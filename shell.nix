{
  pkgs ? import <nixpkgs> { },
  lib ? pkgs.lib,
  ...
}:
let
  package = (import ./. { inherit pkgs; });
in
pkgs.mkShell {

  inputsFrom = [
    package
  ];
  buildInputs = with package.optional-dependencies; dev ++ rtd;
}
