unit Bark.Vocabulary;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types;

type
  TTokenEntry = record
    Text:  String;
    Score: Single;
    Next:  Integer;
  end;

  TTokens = array of TTokenEntry;

  TVocabulary = class
  const
    HASH_SIZE = 65536;
  private
    FEntries: TTokens;
    FBuckets: TIntegerArray;

    FSize:   Integer;
    FLoaded: Boolean;

    FMaxTokenLen: Integer;

    function Hash(const AText: String): Cardinal; inline;

    procedure AddToHash(ATokenID: Integer);

    function FindToken(const AText: String): Integer;
  public
    constructor Create;
    destructor  Destroy; override;

    function  LoadFromFile(const AFileName: String; AVocabSize: Integer): Boolean;

    procedure LoadFromVocabArray(const AVocabArray: TStringArray);

    function Decode(ATokenID: Integer): String;

    function Encode(const AText: String; AAddBos: Boolean): TIntegerArray;

    property Size:   Integer read FSize;
    property Loaded: Boolean read FLoaded;
  end;

implementation

constructor TVocabulary.Create;
begin
  inherited Create;

  FLoaded := False;
end;

destructor TVocabulary.Destroy;
begin
  FEntries := nil;
  FBuckets := nil;

  inherited;
end;

function TVocabulary.Hash(const AText: String): Cardinal;
begin
  Result := 5381;

  for var c in AText do
    Result := ((Result shl 5) + Result) xor Word(c);

  Result := Result and (HASH_SIZE - 1);
end;

procedure TVocabulary.AddToHash(ATokenID: Integer);
var
  H: Cardinal;
begin
  if Length(FEntries[ATokenID].Text) = 0 then
    Exit;

  H := Hash(FEntries[ATokenID].Text);

  FEntries[ATokenID].Next := FBuckets[H];
  FBuckets[H]             := ATokenID;
end;

function TVocabulary.FindToken(const AText: String): Integer;
var
  H:   Cardinal;
  Idx: Integer;
begin
  if Length(FBuckets) = 0 then
    Exit(-1);

  H := Hash(AText);

  Idx := FBuckets[H];

  while Idx <> -1 do
  begin
    if FEntries[Idx].Text = AText then
      Exit(Idx);

    Idx := FEntries[Idx].Next;
  end;

  Result := -1;
end;

function TVocabulary.LoadFromFile(const AFileName: String; AVocabSize: Integer): Boolean;
var
  F:        file;
  Score:    Single;
  TokenLen: Integer;
  RawBytes: array of Byte;
  TokenStr: UTF8String;
begin
  Result  := False;
  FLoaded := False;

  AssignFile(F, AFileName);
  {$I-}Reset(F, 1);{$I+}

  if IOResult <> 0 then
    Exit;

  try
    FSize := Abs(AVocabSize);

    SetLength(FEntries, FSize);
    SetLength(FBuckets, HASH_SIZE);

    for var i := 0 to HASH_SIZE - 1 do
      FBuckets[i] := -1;

    BlockRead(F, FMaxTokenLen, SizeOf(Integer));

    for var i := 0 to FSize - 1 do
    begin
      BlockRead(F, Score, SizeOf(Single));

      FEntries[i].Score := Score;
      FEntries[i].Next  := -1;

      BlockRead(F, TokenLen, SizeOf(Integer));

      if TokenLen > 0 then
      begin
        SetLength(RawBytes, TokenLen);

        BlockRead(F, RawBytes[0], TokenLen);

        SetString(TokenStr, PAnsiChar(@RawBytes[0]), TokenLen);

        FEntries[i].Text := String(TokenStr);
      end
      else
        FEntries[i].Text := '';

      AddToHash(i);
    end;

    FLoaded := True;
    Result  := True;
  finally
    CloseFile(F);
  end;
end;

procedure TVocabulary.LoadFromVocabArray(const AVocabArray: TStringArray);
begin
  FSize := Length(AVocabArray);

  SetLength(FEntries, FSize);
  SetLength(FBuckets, HASH_SIZE);

  for var i := 0 to HASH_SIZE - 1 do
    FBuckets[i] := -1;

  for var i := 0 to FSize - 1 do
    with FEntries[i] do
    begin
      Text  := AVocabArray[i];
      Score := -i;
      Next  := -1;

      AddToHash(i);
    end;

  FLoaded := True;
end;

function TVocabulary.Decode(ATokenID: Integer): String;
var
  S: String;
begin
  if (ATokenID >= 0) and (ATokenID < FSize) then
  begin
    S := FEntries[ATokenID].Text;

    if (S = '<0x0A>') or (S = 'Ċ') then
      Exit(#13#10);

    SetLength(Result, Length(S));

    for var i := 1 to Length(S) do
    begin
      if (S[i] = 'Ġ') or (S[i] = ' ') then
        Result[i] := ' '
      else if S[i] = 'Ċ' then
        Result[i] := #10
      else
        Result[i] := S[i];
    end;
  end
  else
    Result := '';
end;

function PosEx(const ASubStr, AStr: String; AStart: Integer): Integer;
var
  SubLen: Integer;
  StrLen: Integer;
  Match:  Boolean;
begin
  SubLen := Length(ASubStr);
  StrLen := Length(AStr);

  if (SubLen = 0) or (AStart < 1) or (AStart > StrLen) then
    Exit(0);

  for var i := AStart to StrLen - SubLen + 1 do
  begin
    Match := True;

    for var j := 1 to SubLen do
    begin
      if AStr[i + j - 1] <> ASubStr[j] then
      begin
        Match := False;
        Break;
      end;
    end;

    if Match then
      Exit(i);
  end;

  Result := 0;
end;

function TVocabulary.Encode(const AText: string; AAddBos: Boolean): TIntegerArray;
var
  ID, i:     Integer;
  Count:     Integer;
  BestIdx:   Integer;
  BestScore: Single;
  MergedStr: String;
  StrPiece:  String;
  C:         Char;
  EndTag:    Integer;
  Tag:       String;
begin
  SetLength(Result, (Length(AText) * 3) + 2);
  Count := 0;

  if AAddBos then
  begin
    Result[Count] := 1;
    Inc(Count);
  end;

  if Length(AText) > 0 then
  begin
    i := 1;

    while i <= Length(AText) do
    begin
      if (AText[i] = '<') and (i < Length(AText)) and (AText[i + 1] = '|') then
      begin
        EndTag := PosEx('|>', AText, i);

        if EndTag > 0 then
        begin
          Tag := Copy(AText, i, (EndTag + 2) - i);
          ID  := FindToken(Tag);

          if ID <> -1 then
          begin
            Result[Count] := ID;

            Inc(Count);
            i := EndTag + 2;

            Continue;
          end;
        end;
      end;

      C := AText[i];

      if C = #13 then
      begin
        Inc(i);
        Continue;
      end;

      if C = ' ' then
        StrPiece := 'Ġ'
      else if C = #10 then
        StrPiece := 'Ċ'
      else
        StrPiece := C;

      ID := FindToken(StrPiece);

      if (ID = -1) and (C = ' ') then
        ID := FindToken(' ');

      if ID <> -1 then
      begin
        Result[Count] := ID;
        Inc(Count);
      end
      else
      begin
        Result[Count] := (Ord(C) and $FF) + 3;
        Inc(Count);
      end;

      Inc(i);
    end;

    while True do
    begin
      BestScore := -1e10;
      BestIdx   := -1;

      for var j := 0 to Count - 2 do
      begin
        MergedStr := FEntries[Result[j]].Text + FEntries[Result[j + 1]].Text;
        ID        := FindToken(MergedStr);

        if (ID <> -1) and (FEntries[ID].Score > BestScore) then
        begin
          BestScore := FEntries[ID].Score;
          BestIdx   := j;
        end;
      end;

      if BestIdx = -1 then
        Break;

      MergedStr       := FEntries[Result[BestIdx]].Text + FEntries[Result[BestIdx + 1]].Text;
      Result[BestIdx] := FindToken(MergedStr);

      if (Count - BestIdx - 2) > 0 then
        Move(Result[BestIdx + 2], Result[BestIdx + 1], (Count - BestIdx - 2) * SizeOf(Integer));

      Dec(Count);
    end;
  end;

  SetLength(Result, Count);
end;

end.
