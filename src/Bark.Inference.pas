unit Bark.Inference;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types,
  Bark.Model,
  Bark.Maths,
  Bark.Maths.AVX;

type
  TCandidate = record
    Index: Integer;
    Prob:  Single;
  end;

  TCandidates = array of TCandidate;

  TRunState = record
    X:          TSingleArray;
    Xb, Xb2:    TSingleArray;
    Hb, Hb2:    TSingleArray;
    Q, K, V:    TSingleArray;
    Att:        TSingleArray;
    Logits:     TSingleArray;
    KeyCache:   TSingleArray;
    ValueCache: TSingleArray;
  end;

  TInferenceEngine = class
  private
    FModel:       TTransformerModel;
    FState:       TRunState;
    FCandidates:  TCandidates;

    procedure AllocateBuffers;
    procedure SortCandidatesDescending(ACount: Integer);
  public
    constructor Create(AModel: TTransformerModel);
    destructor  Destroy; override;

    function Forward(AToken: Integer; APos: Integer): PSingle;

    function SampleGreedy(ALogits: PSingle): Integer;

    function Sample(ALogits: PSingle; ATemperature: Single; ATopP: Single; const ARecentTokens: TIntegerArray; ARepetitionPenalty: Single): Integer;

    property State: TRunState read FState;
  end;

implementation

constructor TInferenceEngine.Create(AModel: TTransformerModel);
begin
  inherited Create;

  FModel := AModel;

  AllocateBuffers;
end;

destructor TInferenceEngine.Destroy;
begin
  FState.X          := nil;
  FState.Xb         := nil;
  FState.Xb2        := nil;
  FState.Hb         := nil;
  FState.Hb2        := nil;
  FState.Q          := nil;
  FState.K          := nil;
  FState.V          := nil;
  FState.Att        := nil;
  FState.Logits     := nil;
  FState.KeyCache   := nil;
  FState.ValueCache := nil;
  FCandidates       := nil;

  inherited;
end;

procedure TInferenceEngine.AllocateBuffers;
var
  KvDim: Integer;
begin
  with FModel.Config do
  begin
    KvDim := (Dim * NKVHeads) div NHeads;

    SetLength(FState.X,          Dim);
    SetLength(FState.Xb,         Dim);
    SetLength(FState.Xb2,        Dim);
    SetLength(FState.Hb,         HiddenDim);
    SetLength(FState.Hb2,        HiddenDim);
    SetLength(FState.Q,          Dim);
    SetLength(FState.K,          KvDim);
    SetLength(FState.V,          KvDim);
    SetLength(FState.Att,        NHeads * SeqLen);
    SetLength(FState.Logits,     VocabSize);
    SetLength(FState.KeyCache,   NativeInt(NLayers) * SeqLen * KvDim);
    SetLength(FState.ValueCache, NativeInt(NLayers) * SeqLen * KvDim);

    SetLength(FCandidates, VocabSize);
  end;
end;

procedure TInferenceEngine.SortCandidatesDescending(ACount: Integer);
var
  Temp: TCandidate;
begin
  for var i := 1 to ACount - 1 do
  begin
    Temp := FCandidates[i];

    var j := i - 1;

    while (j >= 0) and (FCandidates[j].Prob < Temp.Prob) do
    begin
      FCandidates[j + 1] := FCandidates[j];
      Dec(J);
    end;

    FCandidates[j + 1] := Temp;
  end;
end;

function TInferenceEngine.Forward(AToken: Integer; APos: Integer): PSingle;
var
  C: TModelConfig;
  W: TModelWeights;
  S: TRunState;

  HeadSize:     Integer;
  KvDim, KvMul: Integer;
  CacheOffset:  Integer;
  HeadOffset:   Integer;
  AttOffset:    Integer;
  KvHead:       Integer;
  Score:        Single;
  AttVal:       Single;
  PKeyRow:      PSingle;
  PValRow:      PSingle;
begin
  C := FModel.Config;
  W := FModel.Weights;
  S := FState;

  HeadSize := C.Dim div C.NHeads;
  KvDim    := (C.Dim * C.NKVHeads) div C.NHeads;
  KvMul    := C.NHeads div C.NKVHeads;

  Move(W.TokenEmbedding[AToken * C.Dim], S.X[0], C.Dim * SizeOf(Single));

  for var l := 0 to C.NLayers - 1 do
  begin
    RMSNorm(@S.Xb[0], @S.X[0], @W.RmsAttWeight[l * C.Dim], C.Dim);

    MatMul(@S.Q[0], @S.Xb[0], W.Layers[l].Wq);
    MatMul(@S.K[0], @S.Xb[0], W.Layers[l].Wk);
    MatMul(@S.V[0], @S.Xb[0], W.Layers[l].Wv);

    ApplyRoPE(@S.Q[0], @S.K[0], APos, C.Dim, HeadSize, C.NHeads, C.NKVHeads);

    CacheOffset := (l * C.SeqLen * KvDim) + (APos * KvDim);

    Move(S.K[0], S.KeyCache[CacheOffset],   KvDim * SizeOf(Single));
    Move(S.V[0], S.ValueCache[CacheOffset], KvDim * SizeOf(Single));

    for var h := 0 to C.NHeads - 1 do
    begin
      HeadOffset := h * HeadSize;
      AttOffset  := h * C.SeqLen;
      KvHead     := h div KvMul;

      for var t := 0 to APos do
      begin
        PKeyRow := @S.KeyCache[(l * C.SeqLen * KvDim) + (t * KvDim) + (KvHead * HeadSize)];
        Score   := 0;

        for var i := 0 to HeadSize - 1 do
          Score := Score + (S.Q[HeadOffset + i] * PKeyRow[i]);

        S.Att[AttOffset + t] := Score / Sqrt(HeadSize);
      end;

      Softmax(@S.Att[AttOffset], APos + 1);

      for var i := 0 to HeadSize - 1 do
        S.Xb[HeadOffset + i] := 0.0;

      for var t := 0 to APos do
      begin
        PValRow := @S.ValueCache[(l * C.SeqLen * KvDim) + (t * KvDim) + (KvHead * HeadSize)];
        AttVal  := S.Att[AttOffset + t];

        for var i := 0 to HeadSize - 1 do
          S.Xb[HeadOffset + i] := S.Xb[HeadOffset + i] + (AttVal * PValRow[i]);
      end;
    end;

    MatMul(@S.Xb2[0], @S.Xb[0], W.Layers[l].Wo);
    Accumulate(@S.X[0], @S.Xb2[0], C.Dim);

    RMSNorm(@S.Xb[0], @S.X[0], @W.RmsFfnWeight[l * C.Dim], C.Dim);
    MatMul(@S.Hb[0],  @S.Xb[0], W.Layers[l].W1);
    MatMul(@S.Hb2[0], @S.Xb[0], W.Layers[l].W3);

    for var i := 0 to C.HiddenDim - 1 do
      S.Hb[i] := SiLU(S.Hb[i]) * S.Hb2[i];

    MatMul(@S.Xb[0], @S.Hb[0], W.Layers[l].W2);
    Accumulate(@S.X[0], @S.Xb[0], C.Dim);
  end;

  RMSNorm(@S.X[0], @S.X[0], @W.RmsFinalWeight[0], C.Dim);
  MatMul(@S.Logits[0], @S.X[0], W.Wcls);

  Result := @S.Logits[0];
end;

function TInferenceEngine.SampleGreedy(ALogits: PSingle): Integer;
var
  VocabSize: Integer;
  MaxVal:    Single;
  MaxIdx:    Integer;
begin
  VocabSize := FModel.Config.VocabSize;
  MaxVal    := ALogits[0];
  MaxIdx    := 0;

  for var i := 1 to VocabSize - 1 do
  begin
    if ALogits[i] > MaxVal then
    begin
      MaxVal := ALogits[i];
      MaxIdx := i;
    end;
  end;

  Result := MaxIdx;
end;

function TInferenceEngine.Sample(ALogits: PSingle; ATemperature: Single; ATopP: Single; const ARecentTokens: TIntegerArray; ARepetitionPenalty: Single): Integer;
var
  VocabSize:      Integer;
  TokId:          Integer;
  Rand:           Single;
  CumulativeProb: Single;
  MaxProb:        Single;
  Count:          Integer;
begin
  VocabSize := FModel.Config.VocabSize;

  if (ARepetitionPenalty <> 1) and (Length(ARecentTokens) > 0) then
    for TokId in ARecentTokens do
      if (TokId >= 0) and (TokId < VocabSize) then
      begin
        if ALogits[TokId] > 0 then
          ALogits[TokId] := ALogits[TokId] / ARepetitionPenalty
        else
          ALogits[TokId] := ALogits[TokId] * ARepetitionPenalty;
      end;

  if ATemperature <= 0 then
    Exit(SampleGreedy(ALogits));

  for var i := 0 to VocabSize - 1 do
    ALogits[i] := ALogits[i] / ATemperature;

  Softmax(ALogits, VocabSize);

  if (ATopP <= 0) or (ATopP >= 1) then
  begin
    Rand           := Random;
    CumulativeProb := 0;

    for var i := 0 to VocabSize - 1 do
    begin
      CumulativeProb := CumulativeProb + ALogits[i];

      if Rand < CumulativeProb then
        Exit(i);
    end;

    Exit(VocabSize - 1);
  end;

  MaxProb := ALogits[0];

  for var i := 1 to VocabSize - 1 do
    if ALogits[i] > MaxProb then
      MaxProb := ALogits[i];

  Count := 0;

  for var i := 0 to VocabSize - 1 do
    if ALogits[i] >= (MaxProb * 0.05) then
    begin
      FCandidates[Count].Index := i;
      FCandidates[Count].Prob  := ALogits[i];

      Inc(Count);
    end;

  SortCandidatesDescending(Count);

  Rand           := Random * ATopP;
  CumulativeProb := 0;

  for var i := 0 to Count - 1 do
  begin
    CumulativeProb := CumulativeProb + FCandidates[i].Prob;

    if Rand <= CumulativeProb then
      Exit(FCandidates[i].Index);
  end;

  Result := FCandidates[0].Index;
end;

end.
