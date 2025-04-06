from openf5_utils.ipa import IPAPhonemizer
import phonemizer

# Initialize our custom phonemizer
openf5_phonemizer = IPAPhonemizer()

# Initialize phonemizer library
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

# Compare outputs
text = "Hello, world!"
print("OpenF5 phonemizer:", openf5_phonemizer.phonemize(text))
print("Phonemizer library:", global_phonemizer.phonemize([text])[0])
