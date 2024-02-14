import os

import numpy
import pandas

import datetime
import time

import random

# %matplotlib inline

import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

class CreditCard:

    def __init__(self, n_customers: int,
                 n_terminals: int,
                 random_state: int = 0):
        
        self.n_customers = n_customers
        self.n_terminals = n_terminals
        self.random_state = random_state
        

    def generate_customer_profiles_table(self):
        
        numpy.random.seed(self.random_state)

        customer_id_properties=[]

        # Generate customer properties from random distributions 
        for customer_id in range(self.n_customers):
            
            x_customer_id = numpy.random.uniform(0,100)
            y_customer_id = numpy.random.uniform(0,100)
        
            mean_amount = numpy.random.uniform(5,100) # Arbitrary (but sensible) value
            std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
            mean_nb_tx_per_day = numpy.random.uniform(0,4) # Arbitrary (but sensible) value
            
            customer_id_properties.append([customer_id,
                                           x_customer_id, y_customer_id,
                                           mean_amount, std_amount,
                                           mean_nb_tx_per_day])
            
        customer_profiles_table = pandas.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                                    'x_customer_id', 'y_customer_id',
                                                                                    'mean_amount', 'std_amount',
                                                                                    'mean_nb_tx_per_day'])
        
        return customer_profiles_table
        
    def generate_terminal_profiles_table(self):

        numpy.random.seed(self.random_state)

        terminal_id_properties = []

    
        # Generate terminal properties from random distributions 
        for terminal_id in range(self.n_terminals):
            
            x_terminal_id = numpy.random.uniform(0,100)
            y_terminal_id = numpy.random.uniform(0,100)
            
            terminal_id_properties.append([terminal_id,
                                           x_terminal_id, y_terminal_id])
                                       
        terminal_profiles_table = pandas.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                                    'x_terminal_id', 'y_terminal_id'])
        
        return terminal_profiles_table
    

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = numpy.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = numpy.sqrt(numpy.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(numpy.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals


if __name__ == '__main__':
    credit_card = CreditCard(5, 5)
    customers = credit_card.generate_customer_profiles_table()
    terminals = credit_card.generate_terminal_profiles_table()
    terminal_radius_4 = get_list_terminals_within_radius(customers.iloc[4], terminals[['x_terminal_id', 'y_terminal_id']], r = 50)
    # customers['available_terminals']=customers.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=terminals[['x_terminal_id', 'y_terminal_id']], r = 50), axis = 1)