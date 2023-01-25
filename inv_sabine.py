import math
import numpy as np
import itertools


def inv_sabine(t60, room_dim, c):
    """
    given desired t60, (shoebox) room dimension and sound speed,
    computes the absorption coefficient (amplitude) and image source
    order needed.

    parameters
    ----------
    t60: float
        desired t60 (time it takes to go from full amplitude to 60 db decay) in seconds
    room_dim: list of floats
        list of length 2 or 3 of the room side lengths
    c: float
        speed of sound

    returns
    -------
    absorption: float
        the absorption coefficient (in amplitude domain, to be passed to
        room constructor)
    max_order: int
        the maximum image source order necessary to achieve the desired t60
    """

    # finding image sources up to a maximum order creates a (possibly 3d) diamond
    # like pile of (reflected) rooms. now we need to find the image source model order
    # so that reflections at a distance of at least up to ``c * rt60`` are included.
    # one possibility is to find the largest sphere (or circle in 2d) that fits in the
    # diamond. this is what we are doing here.
    R = []
    for l1, l2 in itertools.combinations(room_dim, 2):
        R.append(l1 * l2 / np.sqrt(l1 ** 2 + l2 ** 2))

    V = np.prod(room_dim)  # area (2d) or volume (3d)
    # "surface" computation is diff for 2d and 3d
    if len(room_dim) == 2:
        S = 2 * np.sum(room_dim)
        sab_coef = 12  # the sabine's coefficient needs to be adjusted in 2d
    elif len(room_dim) == 3:
        S = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
        sab_coef = 24

    a2 = sab_coef * np.log(10) * V / (c * S * t60)  # absorption in power (sabine)

    if a2 > 1.0:
        raise ValueError(
            "evaluation of parameters failed. room may be too large for required t60."
        )

    absorption = 1 - np.sqrt(1 - a2)  # convert to amplitude absorption coefficient

    max_order = math.ceil(c * t60 / np.min(R) - 1)

    return absorption, max_order