from PersistentImageCode import *
from VariancePersistCode import *
def PersDiagram(xyz, lifetime=True):
    plt.rcParams["font.family"] = "Times New Roman"
    D,elements=Makexyzdistance(xyz)
    data=ripser(D,distance_matrix=True)
    rips = Rips(maxdim=2)
    rips.transform(D, distance_matrix=True)
    rips.dgms_[0]=rips.dgms_[0]

    # diamonds = FindDiamonds(xyz)
    # diaArr = np.array([diamonds[0][1],diamonds[0][2]])

    # # #Adding the diamonds into the plotting function
    # for i in range(1,len(diamonds)):
    #   a = np.array([diamonds[i][1],diamonds[i][2]])
    #   diaArr = np.vstack((diaArr,a)) #Rips format!!!!

    # rips.dgms_.append(diaArr)

    # print(rips.dgms_) #A vstack of nparray (birth, death) within a nparray(betti dimensions) 

    rips.plot(show=False, lifetime=lifetime, labels=['Connected Components','Holes','Voids','Diamonds'])
    L = plt.legend()
    plt.setp(L.texts, family="Times New Roman")
    plt.rcParams["font.family"] = "Times New Roman"

def GeneratePI(xyz, savefile=True, pixelx=256, pixely=256, myspread=2, bounds={"maxBD": 4.5, "minBD":-0.1}):
    X=VariancePersistv1(xyz, pixelx=256, pixely=256, myspread= 0.5 ,myspecs=bounds, showplot= False)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=True)

    img=X.reshape(pixelx,pixely)
    pim.show(img)
    if savefile==True:
        plt.imsave(xyz+'_img.png',img, cmap=plt.get_cmap("plasma"), dpi=200)

def GeneratePI2(xyz, savefile=True, pixelx=1000, pixely=1000, myspread=2, bounds={"maxBD": 4.5, "minBD":-0.1}):
    X=VariancePersistv2(xyz, pixelx=1000, pixely=1000, myspread= 0.5 ,myspecs=bounds, showplot= False)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=True)

    img=X.reshape(pixelx,pixely)
    pim.show(img)
    if savefile==True:
        plt.imsave(xyz+'_imgh0.png',img, cmap=plt.get_cmap("plasma"), dpi=200)

def GeneratePI3(xyz, savefile=True, pixelx=1000, pixely=1000, myspread=2, bounds={"maxBD": 4.5, "minBD":-0.1}):
    X=VariancePersistv3(xyz, pixelx=1000, pixely=1000, myspread= 0.5 ,myspecs=bounds, showplot= False)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=True)

    img=X.reshape(pixelx,pixely)
    pim.show(img)
    if savefile==True:
        plt.imsave(xyz+'_imgh1.png',img, cmap=plt.get_cmap("plasma"), dpi=200)

def GeneratePI4(xyz, savefile=True, pixelx=1000, pixely=1000, myspread=2, bounds={"maxBD": 4.5, "minBD":-0.1}):
    X=VariancePersistv4(xyz, pixelx=1000, pixely=1000, myspread= 0.5 ,myspecs=bounds, showplot= False)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=True)

    img=X.reshape(pixelx,pixely)
    pim.show(img)
    if savefile==True:
        plt.imsave(xyz+'_imgh2.png',img, cmap=plt.get_cmap("plasma"), dpi=200)

def GeneratePI5(xyz, savefile=True, pixelx=1000, pixely=1000, myspread=2, bounds={"maxBD": 4.5, "minBD":-0.1}):
    X=VariancePersistv5(xyz, pixelx=1000, pixely=1000, myspread= 0.5 ,myspecs=bounds, showplot= False)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=bounds, verbose=True)

    img=X.reshape(pixelx,pixely)
    pim.show(img)
    if savefile==True:
        plt.imsave(xyz+'_imgh5.png',img, cmap=plt.get_cmap("plasma"), dpi=200)



#Persistent image = persistent diagram with a gaussian normal distribution added on each point in the PD, so in PI it's sort of the center of a density 'mountain'. The higher the electronegativity difference,
#the larger the spread (aka variance) of the mountain