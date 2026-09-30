unit Bark.Maths;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types;

function Power(const ABase, AExponent: Double): Double;
procedure Softmax(AValues: PSingle; ASize: Integer);
procedure RMSNorm(AOutVec, AInVec, AWeight: PSingle; ASize: Integer);
procedure ApplyRoPE(Q, K: PSingle; APos: Integer; ADim, AHeadSize, ANHeads, ANKVHeads: Integer);
procedure Accumulate(AOutVec, AInVec: PSingle; ASize: Integer);
function SiLU(X: Single): Single; inline;

implementation

function Power(const ABase, AExponent: Double): Double;
begin
 if AExponent = 0 then
    Exit(1);

  if ABase = 0 then
    Exit(0);
//  begin
//    if AExponent > 0 then
//      Exit(0)
//    else
//      Exit(0); // Infinity
//  end;

  if ABase > 0 then
    Result := Exp(AExponent * Ln(ABase))
  else
  begin
    if Frac(AExponent) = 0 then
    begin
      Result := Exp(AExponent * Ln(Abs(ABase)));

      if Odd(Trunc(AExponent)) then
        Result := -Result;
    end
    else
      Result := 0;
  end;
end;

procedure Softmax(AValues: PSingle; ASize: Integer);
var
  MaxVal: Single;
  Sum:    Single;
begin
  if ASize <= 0 then
    Exit;

  MaxVal := AValues[0];

  for var i := 1 to ASize - 1 do
    if AValues[i] > MaxVal then
      MaxVal := AValues[i];

  Sum := 0;

  for var i := 0 to ASize - 1 do
  begin
    AValues[i] := Exp(AValues[i] - MaxVal);
    Sum        := Sum + AValues[i];
  end;

  for var i := 0 to ASize - 1 do
    AValues[i] := AValues[i] / Sum;
end;

procedure RMSNorm(AOutVec, AInVec, AWeight: PSingle; ASize: Integer);
var
  SumSq: Single;
  Scale: Single;
begin
  SumSq := 0;

  for var i := 0 to ASize - 1 do
    SumSq := SumSq + (AInVec[i] * AInVec[i]);

  Scale := 1 / Sqrt((SumSq / ASize) + 1e-5);

  for var i := 0 to ASize - 1 do
    AOutVec[i] := AInVec[i] * Scale * AWeight[i];
end;

procedure ApplyRoPE(Q, K: PSingle; APos: Integer; ADim, AHeadSize, ANHeads, ANKVHeads: Integer);
var
  HeadOffset:   Integer;
  HeadDimIndex: Integer;

  Freq, Val, Fcr, Fci: Single;
  Q0, Q1, K0, K1:      Single;
begin
  for var h := 0 to ANHeads - 1 do
  begin
    HeadOffset := h * AHeadSize;

    for var i := 0 to (AHeadSize div 2) - 1 do
    begin
      HeadDimIndex := i * 2;

      Freq := 1.0 / Power(10000, HeadDimIndex / AHeadSize);
      Val  := APos * Freq;

      Fcr := Cos(Val);
      Fci := Sin(Val);

      Q0 := Q[HeadOffset + HeadDimIndex];
      Q1 := Q[HeadOffset + HeadDimIndex + 1];

      Q[HeadOffset + HeadDimIndex]     := (Q0 * Fcr) - (Q1 * Fci);
      Q[HeadOffset + HeadDimIndex + 1] := (Q0 * Fci) + (Q1 * Fcr);
    end;
  end;

  for var h := 0 to ANKVHeads - 1 do
  begin
    HeadOffset := h * AHeadSize;

    for var i := 0 to (AHeadSize div 2) - 1 do
    begin
      HeadDimIndex := i * 2;

      Freq := 1.0 / Power(10000, HeadDimIndex / AHeadSize);
      Val  := APos * Freq;

      Fcr := Cos(Val);
      Fci := Sin(Val);

      K0 := K[HeadOffset + HeadDimIndex];
      K1 := K[HeadOffset + HeadDimIndex + 1];

      K[HeadOffset + HeadDimIndex]     := (K0 * Fcr) - (K1 * Fci);
      K[HeadOffset + HeadDimIndex + 1] := (K0 * Fci) + (K1 * Fcr);
    end;
  end;
end;

procedure Accumulate(AOutVec, AInVec: PSingle; ASize: Integer);
begin
  for var i := 0 to ASize - 1 do
    AOutVec[i] := AOutVec[i] + AInVec[i];
end;

function SiLU(X: Single): Single;
begin
  Result := X / (1 + Exp(-X));
end;

end.
