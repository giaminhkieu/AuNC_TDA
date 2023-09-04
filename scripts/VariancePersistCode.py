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

   #def Distance(x):
    Distance=np.zeros(shape=(len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            Distance[i][j]=np.sqrt(  ((x[i]-x[j])**2)  + ((y[i]-y[j])**2)  + ((z[i]-z[j]) **2)  )
    return [Distance, element]  

# from elements import ELEMENTS
def VariancePersistv1(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0},electroneg_addition=0.4, electroneg_division=10,B1_buffer=0.5,B2_buffer=0.05, showplot=True):
    D,elements=Makexyzdistance(Filename)

    a=ripser(D,distance_matrix=True,maxdim=2)
    
    #Make the birth,death for h0 and h1
    points=(a['dgms'][0][0:-1,1]) #Points = deaths of betti0
    pointsh1=(a['dgms'][1]) #Points h1 = betti1: [birth, death]
    pointsh2=(a['dgms'][2]) #Points h2 = betti2 [birth, death]

    #diagrams = persistence diagram
    diagrams = rips.fit_transform(D, distance_matrix=True)  

    #Find pair electronegativies
    eleneg=list()

       for index in points:
        c=np.where(np.abs((index-a['dperm2all'])) < 0.00000015)[0]
        eleneg.append(np.abs(ELEMENTS[elements[c[0]]].eleneg - ELEMENTS[elements[c[1]]].eleneg)) #append the abs difference between the eleneg of the FIRST TWO elements (????????) in list c
    
    #Formula (| EN1 - EN2| + .4) / 10

    h0matrix=np.hstack(((a['dgms'][0][0:-1,:], np.reshape(((np.array(eleneg)+electroneg_addition)/electroneg_division ), (np.size(eleneg),1)))))

    buffer0=np.full((a['dgms'][1][:,0].size,1), B1_buffer)
    h1matrix=np.hstack((a['dgms'][1],buffer0))

    buffer1=np.full((a['dgms'][2][:,0].size,1),B2_buffer)
    h2matrix=np.hstack((a['dgms'][2],buffer1)) 


    Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))

    pim = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
    
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

#Original function from Townsend 2020
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

print("Successfully imported VariancePersistCode")
