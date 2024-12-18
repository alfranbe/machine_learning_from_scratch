# ---------------------------------------------------
# --------------- Github data loading --------------
# ---------------------------------------------------
"""
In order to read the IDX files, I will use the function <convert_from_file>.
I has been taken from the <idx2numpy> library. 
Concretely, it is placed inside the python file <converters.py>

You can find it here:  https://github.com/ivanyu/idx2numpy/blob/master/idx2numpy/converters.py   on line 49.

That is where all the code in this file come from.
"""

import numpy as np
import struct
from six.moves import reduce
import operator

# Keys are IDX data type codes.
# Values: (ndarray data type name, name for struct.unpack, size in bytes).
_DATA_TYPES_IDX = {
    0x08: ('ubyte', 'B', 1),
    0x09: ('byte', 'b', 1),
    0x0B: ('>i2', 'h', 2),
    0x0C: ('>i4', 'i', 4),
    0x0D: ('>f4', 'f', 4),
    0x0E: ('>f8', 'd', 8)
}

def _internal_convert_MOD(inp):
    """
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    """
    '''
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    '''

    # Read the "magic number" - 4 bytes.
    try:
        mn = struct.unpack('>BBBB', inp.read(4))
    except struct.error:
        raise FormatError(struct.error)

    # First two bytes are always zero, check it.
    if mn[0] != 0 or mn[1] != 0:
        msg = ("Incorrect first two bytes of the magic number: " +
               "0x{0:02X} 0x{1:02X}".format(mn[0], mn[1]))
        raise FormatError(msg)

    # 3rd byte is the data type code.
    dtype_code = mn[2]
    if dtype_code not in _DATA_TYPES_IDX:
        msg = "Incorrect data type code: 0x{0:02X}".format(dtype_code)
        raise FormatError(msg)

    # 4th byte is the number of dimensions.
    dims = int(mn[3])

    # See possible data types description.
    dtype, dtype_s, el_size = _DATA_TYPES_IDX[dtype_code]

    # 4-byte integer for length of each dimension.
    try:
        dims_sizes = struct.unpack('>' + 'I' * dims, inp.read(4 * dims))
    except struct.error as e:
        raise FormatError('Dims sizes: {0}'.format(e))

    # Full length of data.
    full_length = reduce(operator.mul, dims_sizes, 1)

    # Create a numpy array from the data
    try:
        result_array = np.frombuffer(
            inp.read(full_length * el_size),
            dtype=np.dtype(dtype)
        ).reshape(dims_sizes)
    except ValueError as e:
        raise FormatError('Error creating numpy array: {0}'.format(e))

    # Check for superfluous data.
    if len(inp.read(1)) > 0:
        raise FormatError('Superfluous data detected.')

    return result_array

def convert_from_file_MOD(file):
    """
    Reads the content of file in IDX format, converts it into numpy.ndarray and
    returns it.
    file is a file-like object (with read() method) or a file name.
    """
    #if isinstance(file, six_string_types):
    with open(file, 'rb') as f:
        return _internal_convert_MOD(f)


