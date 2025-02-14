import numpy as np
from collections import Counter
from reedsolo import RSCodec
import zlib


def pad_bits(bits, payload_shape):
  total_bits_needed = np.prod(payload_shape)
  if len(bits) < total_bits_needed:
    bits += [0] * (total_bits_needed - len(bits))
  elif len(bits) > total_bits_needed:
    raise ValueError("Message is bigger than the image")
  return bits

rs = RSCodec(250)

def text_to_bits(text, message_shape):
  """Convert text to a list of ints in {0, 1}"""
  message_shape_prod = np.prod(message_shape)
  text_in_bits = bytearray_to_bits(text_to_bytearray(text))

  message = text_in_bits + [0] * 32
  payload = message
  while len(payload) < message_shape_prod:
    payload += message

  payload = payload[:message_shape_prod]
  return payload


def bits_to_text(bits, message_shape):
  """Convert a list of ints in {0, 1} to text"""
  bits = np.reshape(bits, np.prod(message_shape))

  candidates = Counter()
  for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
    candidate = bytearray_to_text(bytearray(candidate))
    if candidate:
      candidates[candidate] += 1

  # choose most common message
  if len(candidates) == 0:
    raise ValueError('Failed to find message.')

  candidate, count = candidates.most_common(1)[0]
  print(f'Found {count} candidates for message. Choosing most common!')

  return candidate



def bytearray_to_bits(x):
  """Convert bytearray to a list of bits"""
  result = []
  for i in x:
    bits = bin(i)[2:]
    bits = '00000000'[len(bits):] + bits
    result.extend([int(b) for b in bits])

  return result


def bits_to_bytearray(bits):
  """Convert a list of bits to a bytearray"""
  ints = []
  for b in range(len(bits) // 8):
    byte = bits[b * 8:(b + 1) * 8]
    ints.append(int(''.join([str(bit) for bit in byte]), 2))

  return bytearray(ints)


def text_to_bytearray(text):
  """Compress and add error correction"""
  assert isinstance(text, str), "expected a string"
  
  x = zlib.compress(text.encode("utf-8"))
  x = rs.encode(bytearray(x))

  return x


def bytearray_to_text(x):
  """Apply error correction and decompress"""
  try:
    assert isinstance(x, (bytes, bytearray)), "expected bytes or bytearray"

    decoded = rs.decode(x)
    
    if isinstance(decoded, tuple):
      decoded = decoded[0]

    decompressed = zlib.decompress(bytes(decoded))
    return decompressed.decode("utf-8")
  except BaseException:
    return False






# def text_to_bits(text, message_shape):
#   """Convert text to a list of ints in {0, 1}"""
#   message_shape_prod = np.prod(message_shape)

#   result = []
#   for c in text:
#     bits = bin(ord(c))[2:]
#     bits = '00000000'[len(bits):] + bits
#     result.extend([int(b) for b in bits])

#   message = result + [0] * 32
#   payload = message
#   while len(payload) < message_shape_prod:
#     payload += message

#   payload = payload[:message_shape_prod]
#   return payload

# def bits_to_text(bits, message_shape):
#   """Convert a list of ints in {0, 1} to text"""
#   bits = np.reshape(bits, np.prod(message_shape))
#   chars = []

#   for b in range(int(len(bits)/8)):
#     byte = bits[b*8:(b+1)*8]
#     chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))

#   full_message = ''.join(chars)

#   candidates = Counter()
#   for candidate in full_message.split('\x00\x00\x00\x00'):
#     if candidate:
#       candidates[candidate] += 1

#   # choose most common message
#   if len(candidates) == 0:
#     raise ValueError('Failed to find message.')

#   candidate, count = candidates.most_common(1)[0]
#   print(f'Found {count} candidates for message, choosing most common.')

#   return candidate