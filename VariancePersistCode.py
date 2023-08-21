### Variance persist below --------------------------
from ElementsInfo import *
from PersistentImageCode import *
from ripser import Rips
from ripser import ripser
rips = Rips()
from sklearn.base import TransformerMixin
import numpy as np
import collections
from itertools import product
import collections
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import scipy.spatial as spatial
import matplotlib.pyplot as plt
def Makexyzdistance(t):
    element=np.loadtxt(t,dtype=str,usecols=(0,), skiprows=2)
    x=np.loadtxt(t,dtype=float,usecols=(1), skiprows=2)
    y=np.loadtxt(t,dtype=float,usecols=(2),skiprows=2)
    z=np.loadtxt(t,dtype=float,usecols=(3),skiprows=2)

  #  print (x)
   #def Distance(x):
    Distance=np.zeros(shape=(len(x),len(x)))
    for i in range(0,len(x)):
       # Make an array for each atom
        for j in range(0,len(x)):
        #Calculate the distance between every atom

            Distance[i][j]=np.sqrt(  ((x[i]-x[j])**2)  + ((y[i]-y[j])**2)  + ((z[i]-z[j]) **2)  )
    return [Distance, element]  #Output: 2 arrays, 1 with the atoms, each atom corresponds with one whole ARRAY above, storing distances between it and every other atoms

# from elements import ELEMENTS
def VariancePersistv1(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0},electroneg_addition=0.4, electroneg_division=10,B1_buffer=0.5,B2_buffer=0.05, showplot=True):
    #Generate distance matrix and elementlist
    #D is a list where each element is another LIST holding the distance of every atoms to this one
    D,elements=Makexyzdistance(Filename)

    #Generate data for persistence diagram
    a=ripser(D,distance_matrix=True,maxdim=2) #ripser outputs a dictionary, the main subject is the 'dgms' list
    
    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = deaths of betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1: [birth, death]
    pointsh2=(a['dgms'][2]) #Points h2 = betti2 [birth, death]

    #diagrams = a barcode in quantitative/literal form (only includes h0 and h1)
    diagrams = rips.fit_transform(D, distance_matrix=True)  

    #Find pair electronegativies
    eleneg=list()
    #Loop through the whole element list to assign electronegativity values

    ### The key to indexing the features (or at least h0)
    #Each index in points is a death (float) of a betti 1 feature, which is equiv to 1 bond being formed
    #a['dperm2all]: a list where each element is a list holding distances to every atoms
    #So abs index - a['dperm2all]: if the death of a betti 1 feature occurs == a bond formed, look for the distance between 2 atoms that are close enough (<0.000015) to the filtration value and append it
    for index in points:
        # c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #np.where grabs the INDEX of the atoms which has the distance value satisfying the limit
        c=np.where(np.abs((index-a['dperm2all'])) < 0.00000015)[0]
        # print ("idx",index)
        # print("idx - dperm2all: ", index - a['dperm2all'])
        # print("current c:",c)
        # eleC =[]
        # for number in c:
        #   eleC.append(elements[number]) 
        # print ("element of current c:",eleC) #elements = the list of elements from the text file;elements[c[0]] == get the name of the element that has index c[0]
        # print("Two atoms: ",elements[c[0]],elements[c[1]])
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg)) #append the abs difference between the eleneg of the FIRST TWO elements (????????) in list c
    
    #new matrix with electronegativity variance in third row, completely empirical
    #Formula (| EN1 - EN2| + .4) / 10
    # print(diagrams[0][0:-1,:])
    # print ((np.array(eleneg)+.4)/10)


    #h0 matrix = rearranging the 'dgms' data into a matrix
    #Buffer = applying a coefficient matrix on top, set at 0.05 by default
    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+electroneg_addition)/electroneg_division ), (np.size(eleneg),1)))))

    buffer0=np.full((a['dgms'][1][:,0].size,1), B1_buffer)
    h1matrix=np.hstack((a['dgms'][1],buffer0))

    buffer1=np.full((a['dgms'][2][:,0].size,1),B2_buffer)
    h2matrix=np.hstack((a['dgms'][2],buffer1)) #Fix: added h2 matrix

    #combine them
    Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))
    # Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))

    # print(Totalmatrix)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    # print (imgs)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

#Only difference: Totalmatrix of v2 only contains the data for h0
def VariancePersistv2(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=True):
    D,elements=Makexyzdistance(Filename)
    
    #Generate data for persistence diagram
    a=ripser(D,distance_matrix=True,maxdim=2)
    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1
    pointsh2=(a['dgms'][2]) #Points h2 = betti2

    # print(pointsh2)

    diagrams = rips.fit_transform(D, distance_matrix=True)  #diagrams = a barcode in quantitative form (only includes h0 and h1)

    #Find pair electronegativies
    eleneg=list()
    # print("dperm2all",a['dperm2all'])
    for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #append indexes 
        # print ("idx",index)
        # print (c)
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
    
    #new matrix with electronegativity variance in third row, completely empirical
    #Formula (| EN1 - EN2| + .4) / 10
    # print(diagrams[0][0:-1,:])
    # print ((np.array(eleneg)+.4)/10)
    
    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    buffer=np.full((a['dgms'][1][:,0].size,1), 0.05)
    h1matrix=np.hstack((a['dgms'][1],buffer))
    buffer1=np.full((a['dgms'][2][:,0].size,1),0.05)
    h2matrix=np.hstack((a['dgms'][2],buffer1)) #Fix: added h2 matrix

    Totalmatrix=np.vstack((h0matrix))

    # print(Totalmatrix)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    # print (imgs)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

#This one shows h1 and so on
def VariancePersistv3(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=True):
    #Generate distance matrix and elementlist
    #D is a list where each element is another LIST holding the distance of every atoms to this one
    D,elements=Makexyzdistance(Filename)
    
    #Generate data for persistence diagram
    a=ripser(D,distance_matrix=True,maxdim=2)
    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1
    pointsh2=(a['dgms'][2]) #Points h2 = betti2

    diagrams = rips.fit_transform(D, distance_matrix=True)  #diagrams = a barcode in quantitative form (only includes h0 and h1)

    #Find pair electronegativies
    eleneg=list()
    # print("dperm2all",a['dperm2all'])
    for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #append indexes 
        # print ("idx",index)
        # print (c)
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
    
    #new matrix with electronegativity variance in third row, completely empirical
    #Formula (| EN1 - EN2| + .4) / 10
    # print(diagrams[0][0:-1,:])
    # print ((np.array(eleneg)+.4)/10)
    
    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    buffer=np.full((a['dgms'][1][:,0].size,1), 0.05)
    h1matrix=np.hstack((a['dgms'][1],buffer))
    buffer1=np.full((a['dgms'][2][:,0].size,1),0.05)
    h2matrix=np.hstack((a['dgms'][2],buffer1)) #Fix: added h2 matrix

    #combine them
    # Totalmatrix=np.vstack((h0matrix,h1matrix))
    Totalmatrix=np.vstack((h1matrix))

    # print(Totalmatrix)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    # print (imgs)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())


def VariancePersistv4(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=True):
    #Generate distance matrix and elementlist
    #D is a list where each element is another LIST holding the distance of every atoms to this one
    D,elements=Makexyzdistance(Filename)
    
    #Generate data for persistence diagram
    a=ripser(D,distance_matrix=True,maxdim=2)
    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1
    pointsh2=(a['dgms'][2]) #Points h2 = betti2

    diagrams = rips.fit_transform(D, distance_matrix=True)  #diagrams = a barcode in quantitative form (only includes h0 and h1)
    # print(diagrams)
    #Find pair electronegativies
    eleneg=list()

    for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #append indexes 
        # print ("idx",index)
        # print (c)
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
    
    #new matrix with electronegativity variance in third row, completely empirical
    #Formula (| EN1 - EN2| + .4) / 10
    # print(diagrams[0][0:-1,:])
    # print ((np.array(eleneg)+.4)/10)
    
    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    buffer=np.full((a['dgms'][1][:,0].size,1), 0.05)
    h1matrix=np.hstack((a['dgms'][1],buffer))
    buffer1=np.full((a['dgms'][2][:,0].size,1),0.05)
    h2matrix=np.hstack((a['dgms'][2],buffer1)) #Fix: added h2 matrix

    #combine them
    # Totalmatrix=np.vstack((h0matrix,h1matrix))
    Totalmatrix=np.vstack((h2matrix))

    # print(Totalmatrix)
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    # print (imgs)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

#This counts the diamonds
def VariancePersistv5(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=True):
    #Generate distance matrix and elementlist
    #D is a list where each element is another LIST holding the distance of every atoms to this one
    D,elements=Makexyzdistance(Filename)
    
    #Generate data for persistence diagram
    a=ripser(D,distance_matrix=True,maxdim=2)

    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1
    pointsh2=(a['dgms'][2]) #Points h2 = betti2


    diagrams = rips.fit_transform(D, distance_matrix=True)  #diagrams = a barcode in quantitative form (only includes h0 and h1)

    #Find pair electronegativies
    eleneg=list()
    # print("dperm2all",a['dperm2all'])
    for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #append indexes 
        # print ("idx",index)
        # print (c)
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))
    
    #new matrix with electronegativity variance in third row, completely empirical
    #Formula (| EN1 - EN2| + .4) / 10
    # print(diagrams[0][0:-1,:])
    # print ((np.array(eleneg)+.4)/10)
    
    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+.4)/10 ), (np.size(eleneg),1)))))

    #FindDiamonds = function below
    diamonds = FindDiamonds(Filename)

    #Rearranging the format to fit with rips
    diaArr = np.array([diamonds[0][1],diamonds[0][2]])
    for i in range(1,len(diamonds)):
      b = np.array([diamonds[i][1],diamonds[i][2]])
      diaArr = np.vstack((diaArr,b)) #Rips format!!!!
      
    #Appending to 'dgms' for consistency
    a['dgms'].append(diaArr)

    buffer=np.full((a['dgms'][1][:,0].size,1), 0.05)
    h1matrix=np.hstack(((a['dgms'][1],buffer)))
    buffer1=np.full((a['dgms'][2][:,0].size,1),0.05)
    h2matrix=np.hstack((a['dgms'][2],buffer1))

    #Diamond buffer also left at default 0.05
    bufferdia=np.full((a['dgms'][3][:,0].size,1),0.05)
    diamatrix=np.hstack((a['dgms'][3],bufferdia)) 

    #combine them
    Totalmatrix=np.vstack((diamatrix))

    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    # print (imgs)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())


#Original function
def VariancePersist(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=True):
    #Generate distance matrix and elementlist
    D,elements=Makexyzdistance(Filename)
   
    #Generate data for persistence diagram, saved in variable "a"
    a=ripser(D,distance_matrix=True)

    # print("a----",a) #Variable "a" includes an array, 1st element is 'dgms' which holds the birth (1st col) and PERSISTENCE (2nd col) of all betti 0 and 1 features, 
    # element 'dperm2all' aka permanent distance to all holds the arrays of distance to other atoms of every atom
    #Make the birth,PERSISTENCE for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #for h0, aka betti0 features (only the PERSISTENCE because they all born at 0)
    pointsh1=(a['dgms'][1]) #For h1, aka betti1 features (an array, each element = [birth,PERSISTENCE])
    diagrams = rips.fit_transform(D, distance_matrix=True)
    #Find pair electronegativies
    eleneg=list()

    # print("points:",points)

    # print("index 0", points[0])
    # print("dperm2all", a['dperm2all'])
    # print("index - a['dperm2all] ", points[0] - a['dperm2all']) #An array, each element is another array holding index - distance to atom x
    # print("np.where(np.abs((index-a['dperm2all'])) < .00000015)",np.where(np.abs((points[0]-a['dperm2all'])) < .00000015))

    for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < .00000015)[0] #????
        
        # print ("ind:",index)
        # print (c)
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg))

    print(eleneg)
    print('lenght', len(eleneg))
   
   
    h0matrix=np.hstack(((diagrams[0][0:-1,:], np.reshape((((np.array(eleneg)*1.05)+.01)/10 ), (np.size(eleneg),1)))))
    buffer=np.full((diagrams[1][:,0].size,1), 0.05)
    h1matrix=np.hstack((diagrams[1],buffer))
    #print (h0matrix)
    #print (h1matrix)
    #combine them
    Totalmatrix=np.vstack((h0matrix,h1matrix))
    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    print (imgs)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())


#Function for finding diamonds
import matplotlib.pyplot as plot
import gudhi
import pandas as pd

def FindDiamonds(t):
    #Read the file, skipping the first 2 lines and creating a GUDHI rips complex
    df=pd.read_table(t, delim_whitespace=True, names=['a','b','c','d'],skiprows = 2)
    mat = df[['b','c','d']].to_numpy()
    points = mat
    rips = gudhi.RipsComplex(points, max_edge_length=7)

    #Creating a simplex tree which holds all the simplex information
    simplex_tree = rips.create_simplex_tree(max_dimension=3)

    #Getting the simplices in the form of ([vertex1, vertex2], filtration value)
    simp = simplex_tree.get_simplices()

    arr =[]
    #Looking for simplices with 4 vertices (aka tetrahedral)
    for i in simp:
      if len(i[0]) == 4 and 2.8 < i[1] and i[1] < 4:
        arr.append(i)

  #2 tetrahedra sharing 3 vertices will form a diamond
    st = []
    for k in range(len(arr)):
      for j in range(k, len(arr)):
        if k == j: continue
        if len(set(arr[k][0]) & set(arr[j][0])) != 3: continue
        # if abs(arr[k][1] - arr[j][1]) >= 0.0000000001: continue
        ver1 = list((set(arr[k][0]) - set(arr[j][0])))[0]
        ver2 = list((set(arr[j][0]) - set(arr[k][0])))[0]
        coord1 = mat[int(ver1)]
        coord2 = mat[int(ver2)]
        dist = np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 +(coord1[2]-coord2[2])**2)
        if max(arr[k][1],arr[j][1]) < dist:
          st.append(([arr[k][0], arr[j][0]], max(arr[k][1],arr[j][1]),dist)) 
    return st

print("Successfully imported VariancePersistCode")