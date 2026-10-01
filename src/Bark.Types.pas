unit Bark.Types;

{$INCLUDE 'Bark.Options.inc'}

interface

type
  TByteArray    = array of Byte;
  TIntegerArray = array of Integer;
  TStringArray  = array of String;
  TSingleArray  = array of Single;

  TBlockQ8 = packed record
    Scale: Single;
    QS:    array[0..31] of Int8;
  end;

  TBlocksQ8 = array of TBlockQ8;

  TWeightTensor = record
    Blocks: TBlocksQ8;
    InDim:  Integer;
    OutDim: Integer;
  end;

  TLayer = record
    Wq, Wk, Wv, Wo: TWeightTensor;
    W1, W2, W3:     TWeightTensor;
  end;

  TLayers = array of TLayer;

  TModelWeights = record
    TokenEmbedding: TSingleArray;
    RmsAttWeight:   TSingleArray;
    RmsFfnWeight:   TSingleArray;
    RmsFinalWeight: TSingleArray;
    Layers:         TLayers;
    Wcls:           TWeightTensor;
  end;

implementation

end.
