import os



if __name__ == "__main__":

    rootPath = "/rhome/yangchen/shared/CleanMORF/randomOutput/BFS/finalNode/depth3"  #d4 302-6394 d5 6395-171390
    for linkerDir in os.listdir(rootPath):
        os.chdir(rootPath + "/" + linkerDir + "/" + linkerDir + "_deformation")
        os.rename(linkerDir + '-claasified-moments.npy', linkerDir + '-classified-moments.npy')
    print("Done")