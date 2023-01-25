# Codebase for VIWER-S GADOAE Multi
A Linear DNN that takes as input arguments a vector of GCC-PHAT maximum locations from multichannel recordings alongside microphone 
coordinates in order to localize sound sources. The algorithm works for arrays comprising 3-15 microphones.

contact: ulrik.kowalk@jade-hs.de

## Algorithms included for comparison:
* SRP-PHAT [1]
* MUSIC [2]

## Voice Activity Algorithms:
* Denk
* Energy
* MaKo
* None
* testNet


## References:
- [1] J.H. DiBiase, "A high-accuracy, low-latency technique for talker localization in reverberant environments using microphone arrays" PhD thesis, Brown University, 2000
- [2] R. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE Trans on Antennas and Propagation, vol 34, no. 3, pp. 276-280, 1986