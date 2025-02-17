import os
from spleeter.separator import Separator
import soundfile as sf

if __name__ == '__main__':
    # Create a separator with the 2-stem model (vocals and accompaniment)
    separator = Separator('spleeter:2stems')

    # Perform the separation
    separator.separate_to_file('test_c5.wav', 'output/')

    # The output will be two files in the 'output' directory:
    # - 'vocals.wav' (likely contains the more prominent voice)
    # - 'accompaniment.wav' (likely contains the background or less prominent voice)

    # Load the separated audio
    vocals, sr_vocals = sf.read('output/comb7/vocals.wav')
    accompaniment, sr_accompaniment = sf.read('output/comb7/accompaniment.wav')

    # Now, amplify Person 1's track (assuming it's 'vocals.wav')
    amplified_vocals = vocals * 1.5

    # Save the processed audio
    sf.write('person1_amplified_7.wav', amplified_vocals, sr_vocals)


