import openbabel
import os
import sys


# start = sys.argv[1]
# end = sys.argv[2]
obConversion = openbabel.OBConversion()
# obConversion.SetInAndOutFormats("xyz", "pdb")
obConversion.SetInAndOutFormats("xyz", "smi")
rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/testBFS/finalNode/depth1"  #d4 302-6394 d5 6395-171390
for linkerDir in os.listdir(rootPath):
# for i in range(int(start),int(end)):
#     linkerDir = "linker" + str(i)
    os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, linkerDir + "-un-squashed.xyz")   # Open Babel will uncompress automatically
    mol.AddHydrogens()
    # obConversion.WriteFile(mol, linkerDir + "-un-squashed.pdb")
    obConversion.WriteFile(mol, linkerDir + ".smi")



print("Done")