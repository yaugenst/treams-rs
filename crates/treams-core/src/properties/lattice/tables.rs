//! Tables of pinned sums, one per line, which [`super::ewald::pinned`] parses.

/// Sums at explicit splits against the automatic split, one per line (see [`pinned`] and
/// `explicit_splits_agree_with_the_automatic_split_or_fail`).
///
/// [`pinned`]: super::ewald::pinned
pub(super) const SPLITS: &str = "
    # Large |k| L: 3D cells whose reciprocal shells reach beyond the base limit.
    s 1 0 3 40.0 0.1 0.1 0.2 0.3 0.3999987500058593 -0.0009999968750146484 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.1 0.2 0.3 value 7e-13: # eta = 0.4 conj(k) / |k|
    # Splits turned 44.5 degrees off 1 / k at |k| = 12, whose reciprocal terms fall 57 times
    # slower than at a real `k eta` of the same modulus: their shells reach beyond
    # |k| sqrt(1 + 100 |eta|^2).
    s 3 -1 2 12.0 0.0 0.6 0.2 0.0 0.14122358893252795 0.13878003433137048 1.7 0.0 0.3 1.8 0.4 -0.3 jet 1e-11:
    c 6 2 12.0 0.0 0.3 0.2 0.0 0.14122358893252795 0.13878003433137048 1.7 0.0 0.4 1.5 0.3 0.1 jet 1e-11:
    # 1D spherical sums with `k eta` turned 43 and 44 degrees, whose reciprocal parts pass
    # |k| sqrt(1 + 100 |eta|^2): their spectral series lie 2e-15 from the automatic split,
    # their Ewald sums continued further 3e-13 and 9e-14.
    s 3 -3 1 28.521978844618726 6.176585967282654 0.02001880451638316 -0.025624426807067557 1.2771687160728806 1.804452784937586 1.1053173712784674 0.9066767318840658 -0.6838183470292138 value 2e-14:
    s 1 -1 1 26.526788666910697 11.871744486836688 -0.04190780764154001 0.05037043945909736 0.12489450116952705 0.9717448891356406 0.3528006401750912 2.8295918313780475 0.1539633715752038 value 2e-14:
    # Splits with |k eta| far below the automatic one, whose real-space shells reach beyond
    # the base limit (k eta = 0.25 and 0.09, off and at a lattice point).
    s 2 1 3 0.0 0.5 0.1 0.2 0.3 0.0 -0.5 1.3 0.0 0.0 0.2 1.5 0.0 0.1 -0.1 1.7 0.1 0.2 -0.3 jet 1e-11:
    s 0 0 3 0.0 0.5 0.0 0.0 0.0 0.0 -0.5 1.3 0.0 0.0 0.2 1.5 0.0 0.1 -0.1 1.7 0.1 0.2 -0.3 jet 1e-11:
    s 2 1 2 0.15 0.03 0.1 0.2 0.3 0.6 0.0 1.3 0.0 0.2 1.5 0.1 0.2 jet 1e-11:
    s 0 0 2 0.15 0.03 0.0 0.0 0.0 0.6 0.0 1.3 0.0 0.2 1.5 0.1 0.2 jet 1e-11:
    # Below every automatic split, turned 84 to 89.99 degrees off 1 / k in (k eta)^2, where
    # the far real-space terms peak up to hundreds of periods out: off the axis the Ewald sums
    # or, beyond 89.9 degrees, the spectral series; on the axis the Ewald sums up to 89.9
    # degrees; and a split turned 89.966 degrees off the axis.
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19429954671295033 0.03912338423423014 0.7289059561347556 -1.366363241200485 scales 1e-8: # 84 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19240807854577835 0.04756137463445268 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19219872197200794 0.048400457840024336 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.5 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19202860190496468 0.04907106170872786 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.9 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19199859027268568 0.049188356143487336 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.97 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19199000236074004 0.04922186546973016 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.99 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19429954671295033 0.03912338423423014 0.7289059561347556 -1.366363241200485 scales 1e-8: # 84 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19240807854577835 0.04756137463445268 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19219872197200794 0.048400457840024336 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.5 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19202860190496468 0.04907106170872786 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.9 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19199859027268568 0.049188356143487336 0.7289059561347556 -1.366363241200485 fails jet: did not converge # 89.97 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 0.0 0.0 0.15861164022252078 0.19199000236074004 0.04922186546973016 0.7289059561347556 -1.366363241200485 fails jet: did not converge # 89.99 deg
    s 10 0 1 2.055764593477441 1.2165207865621608 -0.043824092794076945 0.024071744655843566 0.15861164022252078 0.19200049649728862 0.04918091491210621 0.7289059561347556 -1.366363241200485 scales 1e-8: # 89.966 deg
    # Small |k| below every automatic split: the real-space shells reach their limit, but
    # their Bloch phases cancel what the last two add, and the complete sums (values and
    # jets, which settle separately) are verified against the automatic split; the
    # real-space parts alone fail.
    s 4 -1 2 0.29897784570548896 0.024743641148885533 -0.28498200226849474 -0.029011982202497857 0.24 0.2150988086799845 0.0 0.8 0.0 0.11037932615025436 0.8628889907105753 -3.467326958001231 -1.4014209708794192 value 1e-12:
    s 4 -1 2 0.29897784570548896 0.024743641148885533 -0.28498200226849474 -0.029011982202497857 0.24 0.2150988086799845 0.0 0.8 0.0 0.11037932615025436 0.8628889907105753 -3.467326958001231 -1.4014209708794192 jet 1e-12:
    s 4 -1 2 0.29897784570548896 0.024743641148885533 -0.28498200226849474 -0.029011982202497857 0.24 0.2150988086799845 0.0 0.8 0.0 0.11037932615025436 0.8628889907105753 -3.467326958001231 -1.4014209708794192 fails real: did not converge within the shell limit; use a larger split
    s 7 -3 2 0.20034569239060623 0.0 0.3959688456884133 0.38764428981495824 0.4 0.1695451560326192 0.0 1.0 0.0 0.2 1.1 -1.7730209332374476 -0.4017536791446624 value 1e-12:
    s 7 -3 2 0.20034569239060623 0.0 0.3959688456884133 0.38764428981495824 0.4 0.1695451560326192 0.0 1.0 0.0 0.2 1.1 -1.7730209332374476 -0.4017536791446624 jet 1e-12:
    s 7 -3 2 0.20034569239060623 0.0 0.3959688456884133 0.38764428981495824 0.4 0.1695451560326192 0.0 1.0 0.0 0.2 1.1 -1.7730209332374476 -0.4017536791446624 fails real: did not converge within the shell limit; use a larger split
    s 5 3 3 0.494 0.0 0.049 0.017 0.068 0.1934 0.0 1.0 0.0 0.0 0.1 1.0 0.0 0.1 -0.1 1.0 1.868 2.193 -2.269 value 1e-12:
    # A split turned far off 1 / k above every automatic one keeps the real-space shell
    # limit, whose far shells cancel: it fails, or matches the automatic split.
    c 11 1 -0.9102894564698165 0.9095616163157862 0.9228802242755492 -0.9 0.0 0.6077830589050118 0.0 1.3627151305082263 2.9198840355581073 maybe 1e-10:
    c 11 1 -0.9102894564698165 0.9095616163157862 0.9228802242755492 -0.9 0.0 -0.6077830589050118 0.0 1.3627151305082263 2.9198840355581073 maybe 1e-10:
    c 11 1 -0.9102894564698165 0.9095616163157862 0.9228802242755492 -0.9 0.0 0.9 0.0 1.3627151305082263 2.9198840355581073 maybe 1e-10:
    # Splits whose parts cancel: kept where the loss they predict is small, failing with
    # split too small beyond.
    s 11 7 2 1.0 0.41 0.31 0.17 0.22 0.12 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-12:
    s 11 7 2 1.0 0.41 0.31 0.17 0.22 0.18 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-10:
    s 4 -2 2 1.0 0.41 0.31 0.17 0.22 0.12 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 5e-7:
    s 4 -2 2 1.0 0.41 0.31 0.17 0.22 0.18 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-10:
    c -11 2 1.0 0.41 0.31 0.17 0.0 0.12 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-12:
    c -11 2 1.0 0.41 0.31 0.17 0.0 0.18 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-10:
    c 3 2 1.0 0.41 0.31 0.17 0.0 0.12 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 fails sum: split too small
    c 3 2 1.0 0.41 0.31 0.17 0.0 0.18 0.0 1.0 0.0 0.3 1.1 0.2 -0.3 value 1e-10:
    s 2 1 2 5.0 0.0 0.31 0.17 0.22 0.1 0.0 1.0 0.0 0.0 1.0 0.3 -0.2 fails sum: split too small
    s 2 1 2 5.0 0.0 0.31 0.17 0.22 0.11 0.0 1.0 0.0 0.0 1.0 0.3 -0.2 fails sum: split too small
    # Next to a lattice point at a zero Bloch vector, where the real-space terms add in
    # phase and each part keeps the rounding of its compensated sum.
    s 0 0 3 1.267216528351176 0.0 -9.368968697897542e-5 -5.963162477517018e-6 -3.4448269144448136e-5 0.13 0.0 0.9785488446321685 0.0 0.0 -0.184589817889412 0.8062442062458293 0.0 0.16456364845188753 0.17393727628674394 0.8357180212769595 0.0 0.0 0.0 value 1e-4: # 1e-4 along (-0.9364, -0.0596, -0.3443)
    s 0 0 2 0.6649252047898351 0.0 -7.273182713808302e-6 6.863003221007559e-6 0.0 0.12 0.0 0.8097477035023664 0.0 0.17385500399637133 0.8954293680703219 0.0 0.0 value 1e-4: # 1e-5 along (-0.727, 0.686, 0)
    # A 1D spherical sum without its spectral series (Re k = 0) whose automatic split
    # cancels beyond use.
    s 7 1 1 0.0 1.0 2.5 0.0 0.2 0.0 0.0 1.0 0.3 fails sum: lost its accuracy to cancelling Kambe integrals; reduce the split
    s 7 1 1 0.0 1.0 2.5 0.0 0.2 0.0 0.0 1.0 0.3 fails jet: lost its accuracy to cancelling Kambe integrals; reduce the split
";

/// Sums that vanish by symmetry (see [`pinned`] and
/// `sums_that_vanish_by_symmetry_are_exactly_zero`).
///
/// [`pinned`]: super::ewald::pinned
pub(super) const ZEROS: &str = "
    # Odd degrees half a lattice vector from a lattice point at a zero Bloch vector, at
    # the automatic split and below every automatic one, where the parts cancel.
    c 7 1 1.3 0.4 0.5 0.0 0.0 0.2 0.0 1.0 0.0 jet 1e-6:
    c 7 1 1.3 0.4 0.5 0.0 0.0 0.17 0.0 1.0 0.0 jet 1e-6:
    c 7 1 1.3 0.4 0.5 0.0 0.0 0.155 0.0 1.0 0.0 jet 1e-6:
    c 5 1 1.3 0.4 0.5 0.0 0.0 0.17 0.0 1.0 0.0 jet 1e-6:
    s 7 0 1 0.5 0.0 0.0 0.0 0.5 0.3 0.0 1.0 0.0 jet 1e-6:
    s 5 0 1 0.5 0.0 0.0 0.0 0.5 0.19 0.0 1.0 0.0 jet 1e-6:
    s 15 0 1 0.5 0.0 0.0 0.0 0.5 0.0 0.0 1.0 0.0 jet 1e-6:
    s 3 -3 2 2.7 0.0 0.5 0.0 0.0 0.14 0.0 1.0 0.0 0.3 1.1 0.0 0.0 jet 1e-6:
    c 5 2 2.7 0.0 0.5 0.5 0.0 0.155 0.0 1.0 0.0 0.0 1.0 0.0 0.0 jet 1e-6:
    s 5 1 3 4.1 0.1 0.5 0.0 0.0 0.2 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 jet 1e-6:
    s 1 0 3 2.1 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 jet 1e-6:
    s 1 0 3 2.1 0.0 0.0 0.0 0.0 0.25 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 jet 1e-6:
    # The oblique cell below every automatic split, where the parts cancel beyond use.
    s 3 -3 2 2.7 0.0 0.5 0.0 0.0 0.12 0.0 1.0 0.0 0.3 1.1 0.0 0.0 fails: split too small
    # Odd under a reflection in the lattice axis or plane, at splits turned 90 degrees or
    # more off 1 / k, where the Gaussians of both Ewald parts grow.
    s 9 1 1 0.4 0.8834168701558347 0.0 0.0 0.21877540137439386 0.16851115208178308 0.0 0.8 0.0 fails: non-finite Ewald summand
    s 10 1 2 0.43981783310608696 0.5795441629778919 0.11635568823510342 0.2612745332531094 0.0 0.24216375301902493 0.036599471618345446 1.175377108374894 0.0 0.2242269431819549 0.9791312155659144 -0.4433928811394722 0.8430427352441446 fails: non-finite Ewald summand
";

/// Pinned inputs of the properties (see [`pinned`] and
/// `recorded_cases_keep_their_properties`).
///
/// [`pinned`]: super::ewald::pinned
pub(super) const RECORDED: &str = "
    # Lattice points at the sheets of k eta beyond those of
    # `lattice_point_sums_take_the_sheet_of_k_eta`: a chain at k = -0 + i shifted off its
    # axis, Re k < 0 at a real split turned off 1 / k and at Im k = 0.9, and imaginary
    # splits of modulus 0.3 with either sign of a zero real part.
    s 0 0 1 -0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.7 0.3 point 0.6 0.8 0.0:
    c 0 2 -1.2 0.3 0.0 0.0 0.0 0.6860466044888691 0.13906853155654283 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0: # 0.7 e^(0.2 i)
    c 0 2 -1.2 0.9 0.0 0.0 0.0 0.0 0.0 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0:
    c 0 2 -1.2 0.9 0.0 0.0 0.0 0.7 0.0 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0:
    c 0 2 -1.2 0.9 0.0 0.0 0.0 0.6860466044888691 0.13906853155654283 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0: # 0.7 e^(0.2 i)
    c 0 2 0.0 1.0 0.0 0.0 0.0 -0.0 -0.3 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0:
    s 0 0 1 0.0 1.0 0.0 0.0 0.0 -0.0 -0.3 1.7 0.3 point 0.6 0.8 0.0:
    c 0 2 0.0 1.0 0.0 0.0 0.0 0.0 -0.3 1.7 0.0 0.0 1.5 0.3 0.1 point 0.6 0.8 0.0:
    s 0 0 1 0.0 1.0 0.0 0.0 0.0 0.0 -0.3 1.7 0.3 point 0.6 0.8 0.0:
    # A recorded chain at Im k < 0 now tests rejection of gain media.
    s 0 0 1 0.8 -0.15464196794942472 0.0 0.0 0.0 0.0 0.0 1.3 0.0 point 0 0 0.46436432068201805:
    # A 1D cylindrical sum on its axis whose value and jet paths stop their shells on
    # different components, within the shells they leave out.
    c 8 1 3.8974 2.1474 0.762 0.0 0.0 1.2082 0.0 1.7127 -0.9536 normal 0 1 0:
    # A 2D jet of degree 9 at a split 3.7 times the automatic one, turned 37 degrees off
    # 1 / k: its reciprocal terms fall 0.27 times as fast as at a real `k eta` of the same
    # modulus and reach beyond |k| sqrt(1 + 100 |eta|^2).
    s 9 5 2 3.8934456148495555 2.9295482617625925 1.064233523220655 0.16900349879218846 0.0 1.294840711821749 0.0 1.3 0.0 0.0 1.6900349879218846 -1.9366714370780163 -1.4897164178223457 normal 0 0 1:
    # A 2D jet of degree 9 at an explicit split whose reciprocal terms of dS/da_0x reach
    # 9e3 for a part of 66 (see `check_tiny_normal_shift`).
    s 9 1 2 3.6967030125754947 1.6792835845807719 0.7086231213752138 0.6217780392741852 0.0 1.2310069444183365 0.0 1.812186539561217 0.0 0.0 1.486578616712043 0.0 0.0 normal 0 0 1 1e-20:
    # A 2D jet of degree 15 at |k| = 25 whose reciprocal terms rise like |q + G|^15 from
    # 1e-16 of the sum before they fall: summed past their peak.
    s 15 -15 2 20.668530073483108 14.776134305509029 -0.009652061008259438 -0.7272160178079756 0.4890158736308229 0.0 0.0 1.7778501425657052 0.0 0.21098520175804525 1.7711027756652078 -1.2290420839531804 -0.04299349375290918 direct 1e-12:
    # The chain of SPLITS without its spectral series at a split where it keeps its
    # accuracy.
    s 7 1 1 0.0 1.0 2.5 0.0 0.2 0.0 -0.3 1.0 0.3 direct 1e-12:
    # A degree-9 sum 0.16 from an image whose angular factor lies 8e-6 in cos theta from
    # a zero, where the rounding of the image's unit vector moves the sum by 4.4e-12.
    s 9 4 2 1.0 0.8 0.11188049309921766 0.045194379710055625 0.11147646844645066 0.0 0.0 1.660586582243405 0.0 0.3385394751538423 1.4189428465983411 0.0 0.0 direct 5e-13:
    # A cylindrical chain 5.8e-4 from a lattice point, where a central difference at
    # h = 1e-5 is 5.1e-5 off; 3e-5 from it, where the extrapolation at h = 1e-5 is 5.7e-3
    # off; and
    # at it, where any step in the shift brings back the image the sum leaves out.
    c 0 1 0.8 0.1 -5.8e-4 0.0 0.0 0.7 0.0 1.3 0.0 derivative:
    c 0 1 0.8 0.1 -3e-5 0.0 0.0 0.7 0.0 1.3 0.0 derivative:
    c 0 1 0.8 0.1 0.0 0.0 0.0 0.7 0.0 1.3 0.0 derivative:
    # Sums 0.13 and 0.18 from an image, 1e-20 off a plane where they vanish, which
    # change there by far more than the tolerance: the shrunk input of seed f91a36d1
    # and a degree-12 sum.
    s 9 2 2 0.8 0.0 0.13 0.13 0.0 0.0 0.0 1.3 0.0 0.0 1.3 0.0 0.0 normal 0 0 1 1e-20:
    s 12 1 2 0.8 0.0 0.13 0.0 0.0 0.0 0.0 1.3 0.0 0.0 1.3 0.0 0.0 normal 0 0 1 1e-20:
";
