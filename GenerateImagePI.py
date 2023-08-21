from PersistentImageCode import *
from VariancePersistCode import *
def PersDiagram(xyz, lifetime=True):
    plt.rcParams["font.family"] = "Times New Roman"
    D,elements=Makexyzdistance(xyz)
    data=ripser(D,distance_matrix=True)
    rips = Rips(maxdim=2)
    rips.transform(D, distance_matrix=True)
    rips.dgms_[0]=rips.dgms_[0]

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
