{ pkgs ? import
    (fetchTarball {
      name = "jpetrucciani-2026-02-10";
      url = "https://github.com/jpetrucciani/nix/archive/d4ed47b94e49c6e63eb3667c2cc6c2a45820a824.tar.gz";
      sha256 = "10rg402hpsf5ykzmr3cqdl685rp50b00j0ziv4964w7sbdh163h9";
    })
    { }
}:
let
  name = "skribbl";
  uvEnv = pkgs.uv-nix.mkEnv {
    inherit name; python = pkgs.python313;
    workspaceRoot = pkgs.hax.filterSrc { path = ./.; };
    pyprojectOverrides = final: prev: {
      docopt = prev.docopt.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
      });
      julius = prev.julius.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.setuptools ];
      });
    };
  };

  tools = with pkgs; {
    cli = [
      jfmt
      nixup
      whisper-cpp
      ffmpeg
    ];
    uv = [ uv uvEnv ];

    scripts = pkgs.lib.attrsets.attrValues scripts;
  };

  scripts = with pkgs; { };
  paths = pkgs.lib.flatten [ (builtins.attrValues tools) ];
  env = pkgs.buildEnv {
    inherit name paths; buildInputs = paths;
  };
in
(env.overrideAttrs (_: {
  inherit name;
  NIXUP = "0.0.10";
})) // { inherit scripts; }
