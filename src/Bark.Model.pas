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
    FConfig:  TModelConfig;
    FWeights: TModelWeights;
    FLoaded:  Boolean;

    FVocabulary: TVocabulary;

    function QuantizeFloatToQ8(ASource: PSingle; AInDim, AOutDim: Integer): TWeightTensor;
  public
    constructor Create;
    destructor  Destroy; override;

    function LoadFromBIN(const AFileName: String): Boolean;

    procedure PrintSummary;

    property Config:     TModelConfig  read FConfig;
    property Weights:    TModelWeights read FWeights;
    property Loaded:     Boolean       read FLoaded;
    property Vocabulary: TVocabulary   read FVocabulary;
  end;

implementation

constructor TTransformerModel.Create;
begin
  inherited;

  FLoaded := False;

  FVocabulary := TVocabulary.Create;
end;

destructor TTransformerModel.Destroy;
begin
  FWeights.TokenEmbedding := nil;
  FWeights.RmsAttWeight   := nil;
  FWeights.RmsFfnWeight   := nil;
  FWeights.RmsFinalWeight := nil;
  FWeights.Layers         := nil;
  FWeights.Wcls.Blocks    := nil;

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
    BlockRead(F, FConfig, SizeOf(TModelConfig));

    SharedWeights     := FConfig.VocabSize > 0;
    ActualVocab       := Abs(FConfig.VocabSize);
    FConfig.VocabSize := ActualVocab;
    KvDim             := (FConfig.Dim * FConfig.NKVHeads) div FConfig.NHeads;

    SetLength(FWeights.TokenEmbedding, NativeInt(ActualVocab)     * FConfig.Dim);
    SetLength(FWeights.RmsAttWeight,   NativeInt(FConfig.NLayers) * FConfig.Dim);
    SetLength(FWeights.RmsFfnWeight,   NativeInt(FConfig.NLayers) * FConfig.Dim);
    SetLength(FWeights.RmsFinalWeight, FConfig.Dim);
    SetLength(FWeights.Layers,         FConfig.NLayers);

    BlockRead(F, FWeights.TokenEmbedding[0], Length(FWeights.TokenEmbedding) * SizeOf(Single));
    BlockRead(F, FWeights.RmsAttWeight[0],   Length(FWeights.RmsAttWeight)   * SizeOf(Single));

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].Wq := ReadAndQuantize(FConfig.Dim, FConfig.Dim);

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].Wk := ReadAndQuantize(FConfig.Dim, KvDim);

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].Wv := ReadAndQuantize(FConfig.Dim, KvDim);

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].Wo := ReadAndQuantize(FConfig.Dim, FConfig.Dim);

    BlockRead(F, FWeights.RmsFfnWeight[0], Length(FWeights.RmsFfnWeight) * SizeOf(Single));

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].W1 := ReadAndQuantize(FConfig.Dim, FConfig.HiddenDim);

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].W2 := ReadAndQuantize(FConfig.HiddenDim, FConfig.Dim);

    for var i := 0 to FConfig.NLayers - 1 do
      FWeights.Layers[i].W3 := ReadAndQuantize(FConfig.Dim, FConfig.HiddenDim);

    BlockRead(F, FWeights.RmsFinalWeight[0], FConfig.Dim * SizeOf(Single));

    if not SharedWeights then
      FWeights.Wcls := ReadAndQuantize(FConfig.Dim, ActualVocab)
    else
      FWeights.Wcls := QuantizeFloatToQ8(@FWeights.TokenEmbedding[0], FConfig.Dim, ActualVocab);

    if not FVocabulary.LoadFromFile('tokenizer.bin', FConfig.VocabSize) then
      Exit;

    FLoaded := True;
    Result  := True;
  finally
    CloseFile(F);
    FloatBuf := nil;
  end;
end;

procedure TTransformerModel.PrintSummary;
begin
  Writeln('          Dimensions: ', FConfig.Dim);
  Writeln('   Hidden Dimensions: ', FConfig.HiddenDim);
  Writeln('              Layers: ', FConfig.NLayers);
  Writeln('     Attention Heads: ', FConfig.NHeads);
  Writeln('            KV Heads: ', FConfig.NKVHeads);
  Writeln('     Vocabulary Size: ', Abs(FConfig.VocabSize));
  Writeln(' Max Sequence Length: ', FConfig.SeqLen);
end;


end.
