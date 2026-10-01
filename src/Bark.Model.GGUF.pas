unit Bark.Model.GGUF;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types,
  Bark.Vocabulary,
  Bark.Maths,
  Bark.Model;

type
  TGGUFTensor = record
    Name:       String;
    NDims:      Cardinal;
    Dims:       array[0..3] of UInt64;
    TensorType: Cardinal;
    Offset:     UInt64;
  end;

  TGGUFTensors = array of TGGUFTensor;

  TGGUFReader = class
  const
    GGUF_MAGIC = $46554747;

    GGUF_TYPE_UINT8   = 0;
    GGUF_TYPE_INT8    = 1;
    GGUF_TYPE_UINT16  = 2;
    GGUF_TYPE_INT16   = 3;
    GGUF_TYPE_UINT32  = 4;
    GGUF_TYPE_INT32   = 5;
    GGUF_TYPE_FLOAT32 = 6;
    GGUF_TYPE_BOOL    = 7;
    GGUF_TYPE_STRING  = 8;
    GGUF_TYPE_ARRAY   = 9;
    GGUF_TYPE_UINT64  = 10;
    GGUF_TYPE_INT64   = 11;
    GGUF_TYPE_FLOAT64 = 12;

    GGML_TYPE_F32  = 0;
    GGML_TYPE_F16  = 1;
    GGML_TYPE_Q8_0 = 8;
  private
    FTensors:    TGGUFTensors;
    FAlignment:  Cardinal;
    FDataOffset: UInt64;

    function  ReadString(var F: file): String;
    procedure SkipValue (var F: file; AType: Cardinal);

    function  FindTensor(const AName: String; out ATensor: TGGUFTensor): Boolean;
  public
    constructor Create;
    destructor  Destroy; override;

    function Load(const AFileName: String; AModel: TTransformerModel): Boolean;
  end;

implementation

function IntToStr(Value: Integer): String;
var
  R: ShortString;
begin
  Str(Value, R);
  Result := String(R);
end;

constructor TGGUFReader.Create;
begin
  inherited;

  FAlignment := 32;
end;

destructor TGGUFReader.Destroy;
begin
  FTensors := nil;

  inherited;
end;

function TGGUFReader.ReadString(var F: file): String;
var
  Len:      UInt64;
  RawBytes: TByteArray;
  UStr:     UTF8String;
begin
  BlockRead(F, Len, SizeOf(UInt64));

  if Len > 0 then
  begin
    SetLength(RawBytes, Len);

    BlockRead(F, RawBytes[0], Len);

    SetString(UStr, PAnsiChar(@RawBytes[0]), Len);
    Result := String(UStr);
  end
  else
    Result := '';
end;

procedure TGGUFReader.SkipValue(var F: file; AType: Cardinal);
var
  ItemType: Cardinal;
  Count:    UInt64;
  StrLen:   UInt64;
begin
  case AType of
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL:
      Seek(F, FilePos(F) + 1);

    GGUF_TYPE_UINT16, GGUF_TYPE_INT16:
      Seek(F, FilePos(F) + 2);

    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32:
      Seek(F, FilePos(F) + 4);

    GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64:
      Seek(F, FilePos(F) + 8);

    GGUF_TYPE_STRING:
    begin
      BlockRead(F, StrLen, SizeOf(UInt64));
      Seek(F, FilePos(F) + Integer(StrLen));
    end;

    GGUF_TYPE_ARRAY:
    begin
      BlockRead(F, ItemType, SizeOf(Cardinal));
      BlockRead(F, Count,    SizeOf(UInt64));

      for var i := 0 to Count - 1 do
        SkipValue(F, ItemType);
    end;
  end;
end;

function TGGUFReader.FindTensor(const AName: String; out ATensor: TGGUFTensor): Boolean;
begin
  for var i := 0 to Length(FTensors) - 1 do
    if FTensors[i].Name = AName then
    begin
      ATensor := FTensors[i];

      Exit(True);
    end;

  Result := False;
end;

function TGGUFReader.Load(const AFileName: String; AModel: TTransformerModel): Boolean;
var
  F:           file;
  Magic:       Cardinal;
  Version:     Cardinal;
  ValType:     Cardinal;
  ItemType:    Cardinal;
  TensorCount: UInt64;
  KVCount:     UInt64;
  ArrayCount:  UInt64;
  Key:         String;
  Arch:        String;
  U32Val:      Cardinal;
  CurrentPos:  Int64;
  VocabTokens: TStringArray;
  Info:        TGGUFTensor;

  procedure LoadFloat1D(const Name: String; Dest: PSingle; ExpectedCount: Integer);
  var
    TInfo:   TGGUFTensor;
    HalfBuf: array of UInt16;
  begin
    if not FindTensor(Name, TInfo) then
      Exit;

    Seek(F, FDataOffset + TInfo.Offset);

    if TInfo.TensorType = GGML_TYPE_F32 then
      BlockRead(F, Dest[0], ExpectedCount * SizeOf(Single))

    else if TInfo.TensorType = GGML_TYPE_F16 then
    begin
      SetLength(HalfBuf, ExpectedCount);

      BlockRead(F, HalfBuf[0], ExpectedCount * SizeOf(UInt16));

      for var k := 0 to ExpectedCount - 1 do
        Dest[k] := HalfToFloat(HalfBuf[k]);
    end;
  end;

  function LoadQ8Tensor(const Name: String): TWeightTensor;
  var
    TInfo:         TGGUFTensor;
    TotalElements: Integer;
    NumBlocks:     Integer;
    ScaleH:        UInt16;
  begin
    FillChar(Result, SizeOf(TWeightTensor), 0);

    if not FindTensor(Name, TInfo) then
      Exit;

    Seek(F, FDataOffset + TInfo.Offset);

    Result.InDim := TInfo.Dims[0];

    if TInfo.NDims > 1 then
      Result.OutDim := TInfo.Dims[1]
    else
      Result.OutDim := 1;

    TotalElements := Result.OutDim * Result.InDim;
    NumBlocks     := TotalElements div 32;

    SetLength(Result.Blocks, NumBlocks);

    for var b := 0 to NumBlocks - 1 do
    begin
      BlockRead(F, ScaleH, SizeOf(UInt16));

      Result.Blocks[b].Scale := HalfToFloat(ScaleH);

      BlockRead(F, Result.Blocks[b].QS[0], 32);
    end;
  end;
begin
  Result := False;
  Arch   := 'llama';

  with AModel.Config do
  begin
    Dim       := 576;
    HiddenDim := 1536;
    NLayers   := 30;
    NHeads    := 9;
    NKVHeads  := 9;
    VocabSize := 49152;
    SeqLen    := 2048;
  end;

  AssignFile(F, AFileName);
  {$I-}Reset(F, 1);{$I+}

  if IOResult <> 0 then
    Exit;

  try
    BlockRead(F, Magic, SizeOf(Cardinal));
    if Magic <> GGUF_MAGIC then
      Exit;

    BlockRead(F, Version,     SizeOf(Cardinal));
    BlockRead(F, TensorCount, SizeOf(UInt64));
    BlockRead(F, KVCount,     SizeOf(UInt64));

    for var i := 0 to KVCount - 1 do
    begin
      Key := ReadString(F);

      BlockRead(F, ValType, SizeOf(Cardinal));

      if Key = 'general.architecture' then
        Arch := ReadString(F)

      else if Key = 'general.alignment' then
        BlockRead(F, FAlignment, SizeOf(Cardinal))

      else if Key = (Arch + '.embedding_length') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));
        AModel.Config.Dim := U32Val;
      end

      else if Key = (Arch + '.feed_forward_length') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));
        AModel.Config.HiddenDim := U32Val;
      end

      else if Key = (Arch + '.block_count') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));
        AModel.Config.NLayers := U32Val;
      end

      else if Key = (Arch + '.attention.head_count') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));
        AModel.Config.NHeads := U32Val;
      end

      else if Key = (Arch + '.attention.head_count_kv') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));
        AModel.Config.NKVHeads := U32Val;
      end

      else if Key = (Arch + '.context_length') then
      begin
        BlockRead(F, U32Val, SizeOf(Cardinal));

        if U32Val > 2048 then
          U32Val := 2048;

        AModel.Config.SeqLen := U32Val;
      end

      else if (Key = 'tokenizer.ggml.tokens') and (ValType = GGUF_TYPE_ARRAY) then
      begin
        BlockRead(F, ItemType,   SizeOf(Cardinal));
        BlockRead(F, ArrayCount, SizeOf(UInt64));

        SetLength(VocabTokens, ArrayCount);

        for var j := 0 to ArrayCount - 1 do
          VocabTokens[j] := ReadString(F);

        AModel.Config.VocabSize := ArrayCount;
        AModel.Vocabulary.LoadFromVocabArray(VocabTokens);

        VocabTokens := nil;
      end
      else
        SkipValue(F, ValType);
    end;

    SetLength(FTensors, TensorCount);

    for var i := 0 to TensorCount - 1 do
    begin
      FTensors[i].Name := ReadString(F);

      BlockRead(F, FTensors[i].NDims, SizeOf(Cardinal));

      for var d := 0 to FTensors[i].NDims - 1 do
        BlockRead(F, FTensors[i].Dims[d], SizeOf(UInt64));

      BlockRead(F, FTensors[i].TensorType, SizeOf(Cardinal));
      BlockRead(F, FTensors[i].Offset,     SizeOf(UInt64));
    end;

    CurrentPos := FilePos(F);

    if (CurrentPos mod FAlignment) <> 0 then
      FDataOffset := CurrentPos + (FAlignment - (CurrentPos mod FAlignment))
    else
      FDataOffset := CurrentPos;

    SetLength(AModel.Weights.TokenEmbedding, NativeInt(AModel.Config.VocabSize) * AModel.Config.Dim);
    SetLength(AModel.Weights.RmsAttWeight,   NativeInt(AModel.Config.NLayers)   * AModel.Config.Dim);
    SetLength(AModel.Weights.RmsFfnWeight,   NativeInt(AModel.Config.NLayers)   * AModel.Config.Dim);
    SetLength(AModel.Weights.RmsFinalWeight, AModel.Config.Dim);
    SetLength(AModel.Weights.Layers,         AModel.Config.NLayers);

    if FindTensor('token_embd.weight', Info) then
    begin
      Seek(F, FDataOffset + Info.Offset);

      if Info.TensorType = GGML_TYPE_F32 then
        BlockRead(F, AModel.Weights.TokenEmbedding[0], Length(AModel.Weights.TokenEmbedding) * SizeOf(Single))

      else if Info.TensorType = GGML_TYPE_Q8_0 then
      begin
        var NumBlocks: Int64 := (NativeInt(AModel.Config.VocabSize) * AModel.Config.Dim) div 32;
        var HalfScale: UInt16;
        var QS: array[0..31] of Int8;

        for var blockIdx := 0 to NumBlocks - 1 do
        begin
          BlockRead(F, HalfScale, SizeOf(UInt16));

          var S := HalfToFloat(HalfScale);
          BlockRead(F, QS[0], 32);

          for var j := 0 to 31 do
            AModel.Weights.TokenEmbedding[(blockIdx * 32) + j] := QS[j] * S;
        end;
      end;
    end;

    for var l := 0 to AModel.Config.NLayers - 1 do
    begin
      var LStr := IntToStr(l);

      LoadFloat1D('blk.' + LStr + '.attn_norm.weight', @AModel.Weights.RmsAttWeight[l * AModel.Config.Dim], AModel.Config.Dim);
      LoadFloat1D('blk.' + LStr + '.ffn_norm.weight',  @AModel.Weights.RmsFfnWeight[l * AModel.Config.Dim], AModel.Config.Dim);

      AModel.Weights.Layers[l].Wq := LoadQ8Tensor('blk.' + LStr + '.attn_q.weight');
      AModel.Weights.Layers[l].Wk := LoadQ8Tensor('blk.' + LStr + '.attn_k.weight');
      AModel.Weights.Layers[l].Wv := LoadQ8Tensor('blk.' + LStr + '.attn_v.weight');
      AModel.Weights.Layers[l].Wo := LoadQ8Tensor('blk.' + LStr + '.attn_output.weight');
      AModel.Weights.Layers[l].W1 := LoadQ8Tensor('blk.' + LStr + '.ffn_gate.weight');
      AModel.Weights.Layers[l].W2 := LoadQ8Tensor('blk.' + LStr + '.ffn_down.weight');
      AModel.Weights.Layers[l].W3 := LoadQ8Tensor('blk.' + LStr + '.ffn_up.weight');
    end;

    LoadFloat1D('output_norm.weight', @AModel.Weights.RmsFinalWeight[0], AModel.Config.Dim);

    if FindTensor('output.weight', Info) then
      AModel.Weights.Wcls := LoadQ8Tensor('output.weight')
    else
      AModel.Weights.Wcls := LoadQ8Tensor('token_embd.weight');

    Result := True;
  finally
    CloseFile(F);
  end;
end;

end.
