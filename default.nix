{
  pkgs ? import <nixpkgs> { },
  lib ? pkgs.lib,
  ...
}:
let
  inherit (pkgs) python3Packages;
  inherit (pkgs.lib.fileset) gitTracked intersection toSource;
in
python3Packages.buildPythonPackage {
  format = "pyproject";
  pname = "geoopt";
  version = "5.1.0";
  build-system = [
    python3Packages.flit-core
  ];
  src = toSource {
    root = ./.;
    fileset = gitTracked ./.;
  };
  dependencies = with python3Packages; [
    torch
    scipy
    numpy
  ];

  optional-dependencies = {
    dev = with python3Packages; [
      pytest
      pytest-cov
      pytest-html
      black
      coveralls
      twine
      wheel
      seaborn
      pydocstyle
      pylint
      sphinx
    ];
    rtd = with python3Packages; [
      matplotlib
    ];
  };
}
