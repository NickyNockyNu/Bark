unit Bark.Maths.AVX;

{$INCLUDE 'Bark.Options.inc'}

interface

uses
  Bark.Types;


function DotProduct(AInVec: PSingle; ABlocks: Pointer; ANumBlocks: NativeInt): Single;

procedure MatMul(AOutVec, AInVec: PSingle; const AQW: TWeightTensor);

implementation

function DotProduct(AInVec: PSingle; ABlocks: Pointer; ANumBlocks: NativeInt): Single;
asm
  .NOFRAME
  // RCX = InVec, RDX = Blocks, R8 = NumBlocks
  vxorps  xmm0, xmm0, xmm0
  test    r8, r8
  jz      @Done

@BlockLoop:
  vxorps  ymm1, ymm1, ymm1

  // 0..7
  vmovq       xmm2, qword ptr [rdx + 4]
  vpmovsxbd   ymm2, xmm2
  vcvtdq2ps   ymm2, ymm2
  vmovups     ymm3, [rcx]
  vmulps      ymm2, ymm2, ymm3
  vaddps      ymm1, ymm1, ymm2

  // 8..15
  vmovq       xmm2, qword ptr [rdx + 12]
  vpmovsxbd   ymm2, xmm2
  vcvtdq2ps   ymm2, ymm2
  vmovups     ymm3, [rcx + 32]
  vmulps      ymm2, ymm2, ymm3
  vaddps      ymm1, ymm1, ymm2

  // 16..23
  vmovq       xmm2, qword ptr [rdx + 20]
  vpmovsxbd   ymm2, xmm2
  vcvtdq2ps   ymm2, ymm2
  vmovups     ymm3, [rcx + 64]
  vmulps      ymm2, ymm2, ymm3
  vaddps      ymm1, ymm1, ymm2

  // 24..31
  vmovq       xmm2, qword ptr [rdx + 28]
  vpmovsxbd   ymm2, xmm2
  vcvtdq2ps   ymm2, ymm2
  vmovups     ymm3, [rcx + 96]
  vmulps      ymm2, ymm2, ymm3
  vaddps      ymm1, ymm1, ymm2

  // Reduction & Scale
  vextractf128 xmm2, ymm1, 1
  vaddps       xmm2, xmm2, xmm1
  vhaddps      xmm2, xmm2, xmm2
  vhaddps      xmm2, xmm2, xmm2
  vmulss       xmm2, xmm2, dword ptr [rdx]
  vaddss       xmm0, xmm0, xmm2

  add          rdx, 36
  add          rcx, 128
  dec          r8
  jnz          @BlockLoop

@Done:
  vzeroupper
end;

procedure MatMul(AOutVec, AInVec: PSingle; const AQW: TWeightTensor);
var
  InDim, OutDim, BlocksPerRow: Integer;
  Row: Integer;
  PBlockStart: Pointer;
begin
  InDim        := AQW.InDim;
  OutDim       := AQW.OutDim;
  BlocksPerRow := InDim div 32;

  for Row := 0 to OutDim - 1 do
  begin
    PBlockStart  := @AQW.Blocks[Row * BlocksPerRow];
    AOutVec[Row] := DotProduct(AInVec, PBlockStart, BlocksPerRow);
  end;
end;

end.
