#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 02:52:35 2020

@author: ibrahim
"""

import os
import numpy as np
import pandas as pd


class UCBBandit:
    
    # initialization
    def __init__(self,name,contextSize,):
        self.name = name
        self.A = np.identity(contextSize)
        self.b = np.zeros((contextSize,1))
        self.weights = np.linalg.inv(self.A).dot(self.b)

    def generatePayoff(self,x,alpha = 1):
        x = x.reshape(-1,1)
        payoff = self.weights.flatten().dot(x.flatten()) \
        + alpha * np.sqrt(x.transpose()@np.linalg.inv(self.A)@x)[0,0]
        return(payoff)
                
    def updatePriors(self,outcome,x):
        x = x.reshape(-1,1)
        self.A += x@x.transpose()
        self.b += outcome * x
        
class ThompsonBandit:
    
    # initialization
    def __init__(self,name,contextSize,R = 1,delta = 0.05):
        self.played = 0
        self.vSquare = R **2 * 9 * contextSize* np.log(1/delta)
        self.Bt = np.identity(contextSize)
        self.VectorSum = np.zeros(contextSize)
        self.name = name
        self.weights = np.zeros(shape = (contextSize))
        self.priorMeans = (np.linalg.inv(self.Bt) @ self.VectorSum.reshape(-1,1)).flatten()
        self.covMatrix = self.vSquare * np.linalg.inv(self.Bt)
        
    def generatePayoff(self,x):
        self.weights = np.random.multivariate_normal(self.priorMeans,self.covMatrix)
        return(np.dot(self.weights, x))
                
    def updatePriors(self,outcome,x,contextSize):
        self.played += 1
        self.updateBt(x)
        self.updateVectorSum(x,outcome)  
        self.updateV(contextSize)
        
    def updateBt(self,x):
        x = x.reshape(-1,1)
        self.Bt += x @ np.transpose(x)
        
    def updateVectorSum(self,x,outcome):
        self.VectorSum += x * outcome
    
    def updateV(self,contextSize, R = 1,delta = 0.05):
        self.vSquare = R ** 2 * 9 * contextSize * np.log(self.played/delta)
        
    
def chooseArm(articleIDs,clickProb):
    maxIndices = list(np.argwhere(clickProb == np.amax(clickProb)).flatten())
    if len(maxIndices) == 1:
        winnerID = articleIDs[maxIndices[0]]
    else:
        winnerID = articleIDs[np.random.choice(maxIndices)]
    return winnerID

def gamePlay(history, userHistory,estimatesDict = {}):
   
    if  estimatesDict is  None:
        estimatesDict = {}
        
    rewardHistoryThomps = np.full(history.shape[0],fill_value = -1)
    rewardHistoryUCB =  np.full(history.shape[0],fill_value = -1)
        
    d = userHistory.shape[1] - 1 # d- dimensional feature space
    
    for i in range(len(history)):
        #print("######Iteration Number: {}".format(i))
        period = history.iloc[i,:] #one game-play
        context = userHistory.iloc[i,1:].values
        recommendedID = period[2]  #uniformly sampled article ID for suggestion
        articleIDs = period[2:] #vector of Article IDs available in current period
        articleIDs = articleIDs[articleIDs > 1] #remove the unccessary 0 elements. 1 is the epsilon here
        
        isNotInside = [ i not in list(estimatesDict.keys()) for i in articleIDs] #check whether there is a history of estimates around the article ID
        
        #create the identity A matrix and 0-b vector for Articles that were not updated before
        if sum(isNotInside) > 0: #if all of them already in the dictionary, skip this step 
            estimatesDict.update({ i : [UCBBandit(i,d),ThompsonBandit(i,d)] for i in articleIDs[isNotInside]})
        
        
        # calculate probabilities for each element in the current period #
        clickProbUCB = [estimatesDict[x][0].generatePayoff(context) for x in articleIDs]                    
        chosenArmUCB= chooseArm(articleIDs,clickProbUCB) #chosen by UCB algorithm

        clickProbThompson = [estimatesDict[x][1].generatePayoff(context) for x in articleIDs]                    
        chosenArmThompson= chooseArm(articleIDs,clickProbThompson) #chosen by UCB algorithm
        
        if recommendedID != chosenArmUCB and recommendedID != chosenArmThompson:
            continue
        
        if recommendedID == chosenArmUCB:
            estimatesDict[chosenArmUCB][0].updatePriors(period["click"],context)
            rewardHistoryUCB[i] = period["click"]
        
        if recommendedID == chosenArmThompson:
            estimatesDict[chosenArmThompson][1].updatePriors(period["click"],context,d)
            rewardHistoryThomps[i] = period["click"]
            
            
    
    rewardHistoryUCB = rewardHistoryUCB[rewardHistoryUCB != -1]
    rewardHistoryThomps = rewardHistoryThomps[rewardHistoryThomps != -1]
    rewardUniform =  history.iloc[:len(rewardHistoryUCB),1] #also insert the reward of uniform recommendation policy
    return (rewardHistoryUCB,rewardHistoryThomps,rewardUniform)


# Function to process raw data
def preprocessor(data,stop,start = 0):
    if stop > len(data):
        stop = len(data)
        
    length = stop - start
    dfBatch = data.iloc[start:stop,] #number of Recommendations    
    history = np.empty((length,28)) #initialize empty history of plays #in
    userHistory = np.empty((length,7)) #user-context array
    itemDict = {} #a dictionary for article IDs as keys and features as elements
    
    for i in range(len(dfBatch)): 
        oneEvent = str.split(dfBatch.iloc[i,0],sep = "|") #data if 1 step (1 recommendation game)
        firstComponent = str.split(oneEvent[0], sep = " ") #stamp winnerID - result
        if firstComponent[-1] == "":
            del firstComponent[-1] #delete space element
            
        firstComponent = list(map(int,firstComponent)) #transform to int from str
        
        history[i,0] = firstComponent[0]  #append  timestamp 
        userHistory[i,0] = firstComponent[0] #append  timestamp
        recommendedArticleID = firstComponent[1] 
        history[i,2] = recommendedArticleID
        history[i,1] = firstComponent[2] #append ClickorNot
        
        #append context data
        secondComponent = str.split(oneEvent[1], sep = " ")
        del secondComponent[0]
        
        if secondComponent[-1] == "":
            del secondComponent[-1]
            
        secondComponent = [float(x.split(":")[-1]) for x in secondComponent]
        userHistory[i,1:] = secondComponent #append featurs of the user
        
        #append contexts of articles
        index = 1

        for j in range(2,len(oneEvent)):
            thirdComponent = str.split(oneEvent[j], sep = " ")
            articleID = thirdComponent[0]

            if thirdComponent[-1] == "":
                del thirdComponent[-1]
            
            if articleID not in list(itemDict.keys()):
                itemDict[articleID] = [float(x.split(":")[-1]) for x in thirdComponent[1:]]
            
            if int(articleID)  == recommendedArticleID:
                index = 0 
                continue              
            else:
                history[i,j+index] = int(articleID)
                
    #report results            
    colNamesHistory = ["timestamp", "click", "chosenItem"]
    remainingColumns = ["otherItem{}".format(j) for j in range(1,history.shape[1] - 2)]
    colNamesHistory += remainingColumns
    history = pd.DataFrame(history,columns = colNamesHistory)
    
    colNamesUserHistory = ["timestamp"] + ["feature{}".format(j) for j in range(1,userHistory.shape[1]-1)] + ["constant"]
    userHistory = pd.DataFrame(userHistory,columns = colNamesUserHistory)
    
    return (history, userHistory, itemDict)


##################################### 
             #RESULTS#
##################################### 
    
############## Day 6 ################
fileName = "ydata-fp-td-clicks-v1_0.20090506.gz" #day 6

df  = pd.read_csv(fileName, header = 0, sep = "\n")
         
history,userHistory,_ = preprocessor(df,stop = 6000000, start = 0)

UCB,Thompson,Uniform = gamePlay(history,userHistory) 

#save the individual series
np.savetxt("ucbResults6.csv", UCB, delimiter = ",")
np.savetxt("ThompsonResults6.csv", Thompson, delimiter = ",")
np.savetxt("Uniform6.csv", Uniform, delimiter = ",")

ThompsonReduced = Thompson[:len(UCB)] #equalize the rows
resDay6 = np.concatenate((UCB.reshape(-1,1),ThompsonReduced.reshape(-1,1),Uniform.values.reshape(-1,1)),axis = 1)
resDay6Df = pd.DataFrame(resDay6,columns = ["ucb","thompson","uniform"])
pd.DataFrame.to_csv(resDay6Df,"resDay6.csv") #write it to a csv


###### Plots ######
import seaborn as sns; sns.set()

####### Day 6 ##########
cumulativedf = resDay6Df.cumsum()
cumulativedf = pd.concat([pd.Series(np.array(range(1,cumulativedf.shape[0]+1)),name = "step"), cumulativedf],axis = 1,)
resDay6Df = pd.concat([pd.Series(np.array(range(1,resDay6Df.shape[0]+1)),name = "step"), resDay6Df],axis = 1,)

resDay6Melt = resDay6Df.melt(id_vars='step',
                                   value_vars= list(resDay6Df.columns)[1:],
                                   var_name = "Algorithm",
                                   value_name = "Clicks")

cumulativeMelt = cumulativedf.melt(id_vars='step',
                                   value_vars= list(cumulativedf.columns)[1:],
                                   var_name = "Algorithm",
                                   value_name = "Clicks")

day6CumulativePlot = sns.lineplot(x="step", y="Clicks", style="Algorithm", data=cumulativeMelt,
                                  hue = "Algorithm")
day6RegularPlot = sns.lineplot(x="step", y="Clicks", style="Algorithm", data= resDay6Melt )

##############################################################################
################################# OPTIONAL ###################################
##############################################################################
import math

# function to write the preprocessed data in csv file with preferred batchSize
def writer(history,userHistory, batchSize):
    totalLength = len(history)
    numberOfBatch = math.ceil(totalLength/batchSize)
    for i in range(numberOfBatch):
        fileNameHistory = r'history{}.csv'.format(i+1)
        fileNameUserHistory = r'userHistory{}.csv'.format(i+1) 
        if batchSize * (i+1) > len(history):        
            history.iloc[(batchSize * i):,:].to_csv(fileNameHistory)
            userHistory.iloc[(batchSize * i):,:].to_csv(fileNameUserHistory)
        else:
            history.iloc[(batchSize * i): (batchSize * (i+1)),:].to_csv(fileNameHistory)
            userHistory.iloc[(batchSize * i): (batchSize * (i+1)),:].to_csv(fileNameUserHistory)









