
################ function: greedy_bandits ##################
greedy_bandit <- function(T, theta_truth){
  #number of arms
  K <- length(theta_truth)
  
  #create the prior matrix(keep record of each arm's alpha, beta)
  priors_greedy <- matrix(1,nrow = K, ncol = 2)

  #start with uniform prior (alpha = 1, beta = 1)
  priors_greedy[] <- 1
  
  #create the record of which are is played each round
  record_of_arms <- matrix(data = 0, nrow = T,ncol = K)
  
  #create the record of rewards earned
  record_of_rewards <- vector(mode = "integer", length = T)

  #historical_average_estimates (success probabilities: uniform prior)
  historical_avg = c(0.5,0.5,0.5)
  
  #initialize the previous time t
  t = 1
  
  #start the game
  for (t in seq_len(T)){# loop over the time horizon
    
    if (length(unique(historical_avg)) != 1){
      chosen_arm <- which.max(historical_avg)
      
    } else{#if there is a tie, choose randomly
      chosen_arm <- sample(K, size = 1)
    }
    
    #report which arm is chosen
    record_of_arms[t,chosen_arm] = 1
    
    #draw reward out of the true reward parameter of that arm
    reward <- rbinom(n = 1, size = 1, prob = theta_truth[chosen_arm]) #either 1 or 0
    
    #keep the record of the reward
    record_of_rewards[t] <- reward
    
    #updating the historical averages
    priors_greedy[chosen_arm,] <- priors_greedy[chosen_arm,]  + c(reward,(1-reward))

    historical_avg[chosen_arm] <- priors_greedy[chosen_arm,1]/rowSums(priors_greedy)[chosen_arm]
  }

  result <- list(record_of_arms,record_of_rewards,historical_avg,priors_greedy)
  names(result) <- c("record_of_arms","record_of_rewards","historical_avg","priors_greedy")
  return(result)
}

Greedy_results <- greedy_bandit(T =  10000,theta_truth = c(0.4, 0.5, 0.5))

############# Thompson Sampling ###############
thompson_bandit <- function(T,theta_truth){
  #number of arms
  K <- length(theta_truth)
  
  #create the prior matrix(keep record of each arm's alpha, beta)
  priors <- matrix(1,nrow = K, ncol = 2)
  
  #start with uniform prior (alpha = 1, beta = 1)
  priors[] <- 1
  
  #create the record of which are is played each round
  record_of_arms <- matrix(data = 0, nrow = T,ncol = K)
  
  #create the record of rewards earned
  record_of_rewards <- vector(mode = "integer", length = T)
  
  #initialize the previous time t
  t = 1
  
  #start the game
  for (t in seq_len(T)){# loop over the time horizon
    
    #create the record of probs to be drawn at t
    drawn_prob <- vector(mode = "double",length = K)
    
    #then draw the probabilities for each arm
    for (arm in seq_len(K)){
      draw <- rbeta(1,shape1 = priors[arm,1], shape2 = priors[arm,2])
      drawn_prob[arm] <- draw
    }
    #get the arm with highest prob
    chosen_arm <- which.max(drawn_prob)
    
    #report which arm is chosen
    record_of_arms[t,chosen_arm] = 1
    
    #draw reward out of the true reward parameter of that arm
    reward <- rbinom(n = 1, size = 1, prob = theta_truth[chosen_arm]) #either 1 or 0
    
    #update the priors with the evidence
    priors[chosen_arm,] <- priors[chosen_arm,] + c(reward, 1 - reward)
    
    #keep the record of the reward
    record_of_rewards[t] <- reward
  }
  result <- list(record_of_arms,record_of_rewards,priors)
  names(result) <- c("record_of_arms","record_of_rewards","posteriors")
  return(result)
}

thompson_results <- thompson_bandit(T =  10,theta_truth = c(0.1, 0.2, 0.4))

#perfermance evaluations 

#selecting true arm 10m simulations with T = 1000


#mean reward

#regret





#### IRRELEVANT PART ####
###########Contextual Bandits ############
#each bandit k has its own B_p components such that theta_k = logistic(B,x)

logistic_transform <- function(a){
  1/(1 + exp(a))
}
logistic_transform(5)

#let the true weights of the two bandits are the following
#bandit 0
b_0 <- c(0.1,-2 ,3)

#bandit 1
b_1 <- c(-1, +4,-1)


#generate data
x_1 <- rnorm(n = 1000)
x_2 <- rnorm(n = 1000)
X <- cbind(1,x_1,x_2)

logistic_transform(X%*%b_0)

banit_1_prob <- logistic_transform(X %*% b_0)

#Algorithm 1: ϵ-greedy with regular Logistic Regression

