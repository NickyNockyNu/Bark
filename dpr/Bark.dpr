program Bark;

{$APPTYPE CONSOLE}

uses
  Bark.Types in '..\src\Bark.Types.pas',
  Bark.Vocabulary in '..\src\Bark.Vocabulary.pas';

var
  v: TVocabulary;
  t: TIntegerArray;
  s: String;
begin
  v := TVocabulary.Create;
  v.LoadFromFile('tokenizer.bin', 20000);

  t := v.Encode('I''m sorry, Dave. I''m afraid I can''d do that.', False);

  s := '';

  for var i := 0 to Length(t) - 1 do
    s := s + v.Decode(t[i]);

  Writeln(s);

  Readln;
end.
