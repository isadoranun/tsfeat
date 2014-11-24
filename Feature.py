import os,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import featureFunction



class FeatureSpace:
    """
    This Class is a wrapper class, to allow user select the category of features, or specify a list of features.
    
    __init__ will take in a parameters of category and featureList. 
    User could specify category, which will output all the features tag to this category.
    User could only specify featureList, which will output all the features in the list.
    User could specify category and featureList, which will output all the features in the category and List.
    additional parameters are used for individual features. format is featurename = [parameters]
    
    usage:
    data = np.random.randint(0,10000, 100000000)
    a = FeatureSpace(category='all', automean=[0,0]) # automean is the featurename and [0,0] is the parameter for the feature
    print a.featureList
    a=a.calculateFeature(data)
    print a.result(method='array')
    print a.result(method='dict')

    """
    def __init__(self, category=None, featureList=None, **kwargs):
        self.featureFunc = []
        self.featureList=[]
        
        if category is not None:
            self.category = category
            
            if self.category == 'all':
                for name, obj in inspect.getmembers(featureFunction):
                    if inspect.isclass(obj) and name!='Base':
                        self.featureList.append(name)

            else:
                for name, obj in inspect.getmembers(featureFunction):
                    if inspect.isclass(obj) and name!='Base' :
                        if name in kwargs.keys():
                            if obj( kwargs[name]).category in self.category:
                                self.featureList.append(name)
                            
                        else:
                            if obj().category in self.category:
                                self.featureList.append(name)
            if featureList is not None:
                for item in featureList:
                    self.featureList.append(item)
            
        else:
            self.featureList= featureList
            
        m = featureFunction
        
        for item in self.featureList:
            if item in kwargs.keys():
                try:
                    a = getattr(m, item)( kwargs[item])
                except:
                    print "error in feature "+item
                    sys.exit(1)
            else:
                try:
                    a = getattr(m, item)()
                except:
                    print " could not find feature " + item
                    # discuss -- should we exit?
                    sys.exit(1) 
            try:
                self.featureFunc.append(a.fit)
            except:
                print "could not initilize "+ item


            

        
    def calculateFeature(self, data):
        self._X = np.asarray(data)
        self.__result = []
        for f in self.featureFunc:
            self.__result.append(f(self._X))
        return self
    
    def result(self, method = 'array'):
        if method == 'array':
            return np.asarray(self.__result)
        elif method == 'dict':
            return dict(zip(self.featureList,self.__result))
        elif method == 'features':
            return self.featureList
        else:
            return self.__result

