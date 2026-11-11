# Third-party notices and licensing scope

## Repository licensing scope

- **MIT-licensed verification harness (`LICENSE`):** All Python measurement, synthesis, and plotting code (`scripts/`), the C-ABI benchmark specification (`Sources/NaitDSPBench/include/NaitDSPBench.h`), build scripts (`Makefile`, `requirements.txt`), reference reports (`reports/loopback/`), and documentation (`README.md`) are released under the MIT License.
- **Prebuilt benchmark binary (`lib/libNaitDSPBench.dylib`):** Compiled for macOS `arm64` (Apple Silicon) from the Anchi DAC 64-bit DSP core (`Copyright (c) 2026 Anchizuzu Audio`). Provided free of charge to execute, audit, and reproduce the 26 AES17 and ITU-R verification checks in this repository. Redistribution inside commercial audio products or reverse engineering of proprietary Anchi DAC internals is not granted. Open-source components compiled into `lib/libNaitDSPBench.dylib` are credited below with their full license notices.

---

## 1. `r8brain-free-src` (Aleksey Vaneev, MIT License)

Sample rate converter designed by Aleksey Vaneev of Voxengo (`https://github.com/avaneev/r8brain-free-src`). Used for 64-bit double-precision non-integer rational sample-rate conversion (`44.1 kHz -> 96 / 192 kHz`).

```text
The MIT License (MIT)

r8brain-free-src Copyright (c) 2013-2026 Aleksey Vaneev

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

---

## 2. `PFFFT` double-precision SIMD FFT (Julien Pommier, Hayati Ayguen, Dario Mambro, NCAR / UCAR)

Double-precision ARM NEON / scalar Fourier transform routines bundled inside `r8brain-free-src`, adapted from `FFTPACKv4` by Dr. Paul Swarztrauber (NCAR, 1985).

```text
Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )
Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )

Based on original fortran 77 code from FFTPACKv4 from NETLIB
(http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
of NCAR, in 1985.

As confirmed by the NCAR fftpack software curators, the following
FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
released under the same terms.

FFTPACK license:

http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

Copyright (c) 2004 the University Corporation for Atmospheric
Research ("UCAR"). All rights reserved. Developed by NCAR's
Computational and Information Systems Laboratory, UCAR,
www.cisl.ucar.edu.

Redistribution and use of the Software in source and binary forms,
with or without modification, is permitted provided that the
following conditions are met:

- Neither the names of NCAR's Computational and Information Systems
Laboratory, the University Corporation for Atmospheric Research,
nor the names of its sponsors or contributors may be used to
endorse or promote products derived from this Software without
specific prior written permission.

- Redistributions of source code must retain the above copyright
notices, this list of conditions, and the disclaimer below.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions, and the disclaimer below in the
documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
```

---

## 3. EBU Tech 3253 — Sound Quality Assessment Material (`SQAM`)

Studio acoustic reference recordings cited in the Anchi DAC white paper and supported as optional user-supplied local `.flac` inputs during interactive ear-training exercises (`scripts/loopback_lab/synthesizer.py`) are published by the **European Broadcasting Union (EBU)**, Geneva (`EBU Tech 3253`, `https://tech.ebu.ch/publications/sqamcd`). No EBU audio files are bundled or redistributed in this repository; when local `.flac` masters are absent, the test suite synthesizes mathematical test stems automatically.

---

## 4. Kimiko Ishizaka — *The Open Goldberg Variations* (`BWV 988`, `CC0 1.0`)

High-resolution (`96 kHz / 24-bit`) solo piano reference masters cited in low-level quantization decay analysis are from J. S. Bach's *Open Goldberg Variations* (`BWV 988`), performed by **Kimiko Ishizaka** on a Bösendorfer 290 Imperial at Teldex Studio, Berlin, produced with **Robert Douglass**, and dedicated to the public domain under Creative Commons `CC0 1.0 Universal` (`https://opengoldbergvariations.org/`). No audio recordings are bundled in this repository.
