import matplotlib.pyplot as plt 
import numpy as np

def plot_score(scores, title, limit_score = 13):
    '''
    Gets an array of scores and returns the plot of the scores and the 100 sample moving average
    If the array length is <100 only the scores are plotte
    
    title should be the name of the architecture such as DQN, Double DQN etc.
    '''
    n = len(scores)
              
            
    # plot the scores
    fig = plt.figure(figsize=(10.65,6),  )
    ax = fig.add_subplot(111)
    plt.plot(scores)
    
    if n >100:
        avg_x = []
        avg_y = []
        for i in range(100, n+1):
            avg_x.append(i)
            avg_y.append(np.average(scores[i-100:i]))
        plt.plot(avg_x, avg_y,'r')
    
    plt.axhline(y=limit_score,  linestyle='dotted', color='r')
    
    plt.xlim(0,n)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.grid()
    plt.title("Score over # of episodes for " + title)
    plt.show()
        
        
        
    
    
    
    