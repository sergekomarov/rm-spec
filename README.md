# arm-spec

### Usage
Please see USAGE.ipynb.

### The problem

When one needs to deal with patchy images with complicated edges, straightforward application of a Fourier transform leads to erroneous results (e.g. wrong spectral slope or/and normalization). Several approaches have been proposed in literature. 

### Approaches

A common method to calculate the power spectral density of 2D data with complicated boundaries and/or missing parts is to use the so-called structure function <(f(x+r)-f(x))^2> popular in turbulence studies. The advantage is that the structure function is not sensitive to fluctuations on scales larger that the area under investigation, as opposed to the two-point correlation function <f(x)f(x+r)>. One disadvantage however is that it has a low spectral resolution, so it blurs spectral features significantly. 

### Delta-variance method / Mexican hat filtering

An alternative is to apply a Mexican hat wavelet separately to a patchy image with gaps replaced by zeros and to its 0/1 mask (capturing the gaps and boundaries), divide one by another, and calculate the total variance in the resulting filtered image. By repeating this procedure at different scales, one can estimate the power spectral density. It turns out, this approach leads to much less spectral blurring, while being simple and robust.

Here for an astrophysical example, the method is applied to rotation measure maps to infer the spectrum of magnetic fields in a galaxy cluster. But the code can be used with any 2D data converted to an .npy array or a .fits file with NaNs indicating missing parts.
 
### Papers used:
"A Mexican hat with holes: calculating low-resolution power spectra from data with gaps", P. Arevalo et al., 2012
"Structure analysis of interstellar clouds. I. Improving the Δ-variance method", V. Ossenkopf, M. Krips, and J. Stutzki, 2008
