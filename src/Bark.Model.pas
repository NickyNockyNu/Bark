unit Bark.Model;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types,
  Bark.Vocabulary;

type
  TModelConfig = packed record
    Dim:       Integer;
    HiddenDim: Integer;
    NLayers:   Integer;
    NHeads:    Integer;
    NKVHeads:  Integer;
    VocabSize: Integer;
    SeqLen:    Integer;
  end;

  TTransformerModel = class
  private
    FLoaded:       Boolean;
    FVocabulary:   TVocabulary;
    FRopeFreqBase: Single;

    function QuantizeFloatToQ8(ASource: PSingle; AInDim, AOutDim: Integer): TWeightTensor;
  public
    Config:  TModelConfig;
    Weights: TModelWeights;

    constructor Create;
    destructor  Destroy; override;

    function LoadFromBIN (const AFileName: String): Boolean;
    function LoadFromGGUF(const AFileName: String): Boolean;

    procedure PrintSummary;

    property Loaded:     Boolean       read FLoaded;
    property Vocabulary: TVocabulary   read FVocabulary;

    property RopeFreqBase: Single read FRopeFreqBase write FRopeFreqBase;
  end;

implementation

uses
  Bark.Model.GGUF;

constructor TTransformerModel.Create;
begin
  inherited;

  FLoaded := False;

  FRopeFreqBase := 10000;

  FVocabulary := TVocabulary.Create;
end;

destructor TTransformerModel.Destroy;
begin
  Weights.TokenEmbedding := nil;
  Weights.RmsAttWeight   := nil;
  Weights.RmsFfnWeight   := nil;
  Weights.RmsFinalWeight := nil;
  Weights.Layers         := nil;
  Weights.Wcls.Blocks    := nil;

  FVocabulary.Free;

  inherited;
end;

function TTransformerModel.QuantizeFloatToQ8(ASource: PSingle; AInDim, AOutDim: Integer): TWeightTensor;
var
  Total:     Integer;
  NumBlocks: Integer;
  MaxVal:    Single;
  Val:       Single;
  Scale:     Single;
  Offset:    Integer;
begin
  Result.InDim  := AInDim;
  Result.OutDim := AOutDim;

  Total     := AOutDim * AInDim;
  NumBlocks := Total div 32;

  SetLength(Result.Blocks, NumBlocks);

  for var i := 0 to NumBlocks - 1 do
  begin
    Offset := i * 32;

    MaxVal := 0;

    for var j := 0 to 31 do
    begin
      Val := Abs(ASource[Offset + j]);

      if Val > MaxVal then
        MaxVal := Val;
    end;

    if MaxVal > 0 then
      Scale := MaxVal / 127
    else
      Scale := 1;

    Result.Blocks[i].Scale := Scale;

    for var j := 0 to 31 do
    begin
      Val := ASource[Offset + J];

      Result.Blocks[i].QS[j] := Round(Val / Scale);
    end;
  end;
end;

function TTransformerModel.LoadFromBIN(const AFileName: String): Boolean;
var
  F:             file;
  ActualVocab:   Integer;
  KvDim:         Integer;
  SharedWeights: Boolean;
  FloatBuf:      TSingleArray;

  function ReadAndQuantize(InDim, OutDim: Integer): TWeightTensor;
  var
    Count: Integer;
  begin
    Count := InDim * OutDim;

    if Length(FloatBuf) < Count then
      SetLength(FloatBuf, Count);

    BlockRead(F, FloatBuf[0], Count * SizeOf(Single));

    Result := QuantizeFloatToQ8(@FloatBuf[0], InDim, OutDim);
  end;
begin
  Result  := False;
  FLoaded := False;

  AssignFile(F, AFileName);
  {$I-}Reset(F, 1);{$I+}

  if IOResult <> 0 then
    Exit;

  try
    BlockRead(F, Config, SizeOf(TModelConfig));

    SharedWeights    := Config.VocabSize > 0;
    ActualVocab      := Abs(Config.VocabSize);
    Config.VocabSize := ActualVocab;
    KvDim            := (Config.Dim * Config.NKVHeads) div Config.NHeads;

    SetLength(Weights.TokenEmbedding, NativeInt(ActualVocab)    * Config.Dim);
    SetLength(Weights.RmsAttWeight,   NativeInt(Config.NLayers) * Config.Dim);
    SetLength(Weights.RmsFfnWeight,   NativeInt(Config.NLayers) * Config.Dim);
    SetLength(Weights.RmsFinalWeight, Config.Dim);
    SetLength(Weights.Layers,         Config.NLayers);

    BlockRead(F, Weights.TokenEmbedding[0], Length(Weights.TokenEmbedding) * SizeOf(Single));
    BlockRead(F, Weights.RmsAttWeight[0],   Length(Weights.RmsAttWeight)   * SizeOf(Single));

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].Wq := ReadAndQuantize(Config.Dim, Config.Dim);

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].Wk := ReadAndQuantize(Config.Dim, KvDim);

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].Wv := ReadAndQuantize(Config.Dim, KvDim);

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].Wo := ReadAndQuantize(Config.Dim, Config.Dim);

    BlockRead(F, Weights.RmsFfnWeight[0], Length(Weights.RmsFfnWeight) * SizeOf(Single));

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].W1 := ReadAndQuantize(Config.Dim, Config.HiddenDim);

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].W2 := ReadAndQuantize(Config.HiddenDim, Config.Dim);

    for var i := 0 to Config.NLayers - 1 do
      Weights.Layers[i].W3 := ReadAndQuantize(Config.Dim, Config.HiddenDim);

    BlockRead(F, Weights.RmsFinalWeight[0], Config.Dim * SizeOf(Single));

    if not SharedWeights then
      Weights.Wcls := ReadAndQuantize(Config.Dim, ActualVocab)
    else
      Weights.Wcls := QuantizeFloatToQ8(@Weights.TokenEmbedding[0], Config.Dim, ActualVocab);

    if not FVocabulary.LoadFromFile('tokenizer.bin', Config.VocabSize) then
      Exit;

    FLoaded := True;
    Result  := True;
  finally
    CloseFile(F);
    FloatBuf := nil;
  end;
end;

function TTransformerModel.LoadFromGGUF(const AFileName: String): Boolean;
var
  Reader: TGGUFReader;
begin
  Reader := TGGUFReader.Create;
  try
    FLoaded := Reader.Load(AFileName, Self);
    Result  := FLoaded;
  finally
    Reader.Free;
  end;
end;

procedure TTransformerModel.PrintSummary;
begin
  Writeln('          Dimensions: ', Config.Dim);
  Writeln('   Hidden Dimensions: ', Config.HiddenDim);
  Writeln('              Layers: ', Config.NLayers);
  Writeln('     Attention Heads: ', Config.NHeads);
  Writeln('            KV Heads: ', Config.NKVHeads);
  Writeln('     Vocabulary Size: ', Abs(Config.VocabSize));
  Writeln(' Max Sequence Length: ', Config.SeqLen);
end;


end.
