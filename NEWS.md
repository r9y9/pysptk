# News for pysptk

## v0.1.3

- Building binary wheels for Windows using Appveyor ([#28])
- Add Installation guide on windows ([#25])
- Start Windows continuous integration on AppVeyor ([#24]). As part of the issue, binary dependency was updated so that SPTK library can be compiled on linux, osx and Windows as well.
- Remove unnecesarry array initialization ([#23])

## v0.1.2

- Add `pysptk.synthesis` package that provides high level interfaces for speech waveform synthesis ([#14])
- Add cross-link to the docs
- Add `pysptk.conversion.mgc2b`
- Add speech analysis and re-synthesis demonstration notebook ([#13])
- Add `pysptk.util.example_audio_file`
- Add `fftcep` ([#18])
- Add `mfcc` ([#21])
- Cython is now only required to build development versioni of pysptk. ([#8])

## v0.1.1

- Include *.c to pypi distribution

## v0.1.0

- Initial release

[#8]: https://github.com/r9y9/pysptk/issues/8
[#13]: https://github.com/r9y9/pysptk/issues/13
[#14]: https://github.com/r9y9/pysptk/pull/14
[#18]: https://github.com/r9y9/pysptk/issues/18
[#21]: https://github.com/r9y9/pysptk/pull/21
[#23]: https://github.com/r9y9/pysptk/pull/23
[#24]: https://github.com/r9y9/pysptk/pull/24
[#25]: https://github.com/r9y9/pysptk/issues/25
[#28]: https://github.com/r9y9/pysptk/pull/28