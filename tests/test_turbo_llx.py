#
# Turbo Decoder Unit Tests
#

import unittest

from turbo import RSC
from turbo import TurboEncoder
from turbo import AWGN
from turbo import SISODecoder
from turbo import TurboDecoder


class TestEncoder(unittest.TestCase):
    def test_rsc_encoder(self):
        rsc = RSC()

        input_vector = [1, 1, 0, 1]
        output_vector, _ = rsc.execute(input_vector)

        print("")
        print("--test_rsc_encoder--")
        print("input_vector = {}".format(input_vector))
        print("output_vector = {}".format(output_vector))
        print("state = {}".format(rsc.registers))

        self.assertListEqual(list(rsc.registers), len(rsc.registers) * [0])

    def test_turbo_encoder(self):
        interleaver = [2, 0, 3, 1]
        turbo_encoder = TurboEncoder(interleaver)

        input_vector = [1, 1, 0, 1]
        output_vector = turbo_encoder.execute(input_vector)

        expected_vector_1 = [1, 1, 1, 0]
        expected_vector_2 = [0, 1, 1, 0]

        print("")
        print("--test_turbo_encoder--")
        print("output = {}".format(output_vector))

        self.assertListEqual(list(output_vector[:-6][1::3]), expected_vector_1)
        self.assertListEqual(list(output_vector[:-6][2::3]), expected_vector_2)


class TestTurboDecoder(unittest.TestCase):
    def test_siso_decoder(self):
        interleaver = 10 * [0]
        block_size = len(interleaver) + 2

        encoder = TurboEncoder(interleaver)

        channel = AWGN(5)
        decoder = SISODecoder(block_size)

        input_vector = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        encoded_vector = encoder.execute(input_vector)

        channel_vector = list(map(float, encoded_vector))
        channel_vector = channel.convert_to_symbols(channel_vector)

        channel_vector = channel.execute(channel_vector)
        demultiplexed_vector = decoder.demultiplex(channel_vector)

        decoded_vector = decoder.execute(demultiplexed_vector)
        decoded_vector = [int(b > 0.0) for b in decoded_vector]

        print("")
        print("--test_siso_decoder--")
        print("input_vector = {}".format(input_vector))
        print("encoded_vector = {}".format(list(encoded_vector)))
        print("decoded_vector = {}".format(decoded_vector))

        self.assertListEqual(list(encoded_vector[::3]), decoded_vector)

    def test_turbo_decoder(self):
        interleaver = [5, 4, 1, 3, 11, 0, 8, 7, 2, 9, 10, 6]
        interleaver_flip = [5, 6, 1, 3, 11, 0, 8, 7, 2, 9, 10, 6]
        encoder = TurboEncoder(interleaver)
        decoder = TurboDecoder(interleaver)

        channel = AWGN(20)

        input_vector = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]
        encoded_vector = encoder.execute(input_vector)

        channel_vector = list(map(float, encoded_vector))
        channel_vector = channel.convert_to_symbols(channel_vector)

        channel_vector = channel.execute(channel_vector)

        decoded_vector = decoder.execute(channel_vector)
        decoded_vector = [int(b > 0.0) for b in decoded_vector]

        print("")
        print("--test_turbo_decoder--")
        print("input_vector = {}".format(input_vector))
        print("encoded_vector = {}".format(list(encoded_vector[:-6])))
        print("decoded_vector = {}".format(decoded_vector[:-2]))

        # self.assertListEqual(list(encoded_vector[:-6][::3]), decoded_vector[:-2])
        self.assertListEqual(list(input_vector), decoded_vector[:-2])
        

if __name__ == "__main__":
    unittest.main()
