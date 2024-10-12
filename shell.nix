{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;
  pythonPackages = python.pkgs;
in
pkgs.mkShell {
  packages = with pkgs; [
    ollama
  ];

  buildInputs = [
    python
    pythonPackages.pip
    pythonPackages.virtualenv
  ];

  shellHook = ''
    rm -rf venv/
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  '';
}
