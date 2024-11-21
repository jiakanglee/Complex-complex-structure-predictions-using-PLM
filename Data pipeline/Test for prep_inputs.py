import re
import os

def prep_inputs(sequence, jobname="test", homooligomer="1", output_dir=None, clean=False, verbose=True):
  # process inputs
  sequence = str(sequence)
  sequence = re.sub("[^A-Z:/]", "", sequence.upper())
  sequence = re.sub(":+", ":", sequence)
  sequence = re.sub("/+", "/", sequence)
  sequence = re.sub("^[:/]+", "", sequence)
  sequence = re.sub("[:/]+$", "", sequence)
  jobname = re.sub(r'\W+', '', jobname)
  homooligomer = str(homooligomer)
  homooligomer = re.sub("[:/]+", ":", homooligomer)
  homooligomer = re.sub("^[:/]+", "", homooligomer)
  homooligomer = re.sub("[:/]+$", "", homooligomer)

  if len(homooligomer) == 0: homooligomer = "1"
  homooligomer = re.sub("[^0-9:]", "", homooligomer)

  # define inputs
  I = {"ori_sequence": sequence,
       "sequence": sequence.replace("/", "").replace(":", ""),
       "seqs": sequence.replace("/", "").split(":"),
       "homooligomer": homooligomer,
       "homooligomers": [int(h) for h in homooligomer.split(":")],
       "msas": [], "deletion_matrices": []}

  # adjust homooligomer option
  if len(I["seqs"]) != len(I["homooligomers"]):
    if len(I["homooligomers"]) == 1:
      I["homooligomers"] = [I["homooligomers"][0]] * len(I["seqs"])
    else:
      if verbose:
        print("WARNING: Mismatch between number of breaks ':' in 'sequence' and 'homooligomer' definition")
      while len(I["seqs"]) > len(I["homooligomers"]):
        I["homooligomers"].append(1)
      I["homooligomers"] = I["homooligomers"][:len(I["seqs"])]
    I["homooligomer"] = ":".join([str(h) for h in I["homooligomers"]])

  # define full sequence being modelled
  I["full_sequence"] = ''.join([s * h for s, h in zip(I["seqs"], I["homooligomers"])])
  I["lengths"] = [len(seq) for seq in I["seqs"]]



  return I

print(prep_inputs("PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK:MRWQEMGYIFYPRKLR"))