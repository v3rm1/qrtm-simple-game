import numpy as np
import random
import pandas as pd
from pandas import ExcelWriter


RAND_MAX = 2147483647
########################################
### The Regression Tsetlin Machine #####
########################################

class TsetlinMachine:


	# Initialization of the Regression Tsetlin Machine
	def __init__(self, number_of_clauses, number_of_features, number_of_states, s, threshold, max_target, min_target, logger):
		j = 0

		self.number_of_clauses = number_of_clauses
		self.number_of_features = number_of_features
		self.number_of_states = number_of_states
		self.s = s
		self.threshold = threshold
		self.max_target = max_target
		self.min_target = min_target

		# The state of each Tsetlin Automaton is stored here. The automata are randomly initialized to either 'number_of_states' or 'number_of_states' + 1.
		self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1], size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)

		# Data structure for keeping track of the sign of each clause
		self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
		
		# Data structures for intermediate calculations (clause output, summation of votes, and feedback to clauses)
		self.clause_output = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)
		self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)
		if logger != None:
			self.logger = logger
		# Set up the Regression Tsetlin Machine structure
		for j in range(self.number_of_clauses):
			if j % 2 == 0:
				self.clause_sign[j] = 1
			else:
				self.clause_sign[j] = 1


	# Calculate the output of each clause using the actions of each Tsetlin Automaton.
	def calculate_clause_output(self, X):
		j = 0
		k = 0
		for j in range(self.number_of_clauses):				
			self.clause_output[j] = 1
			for k in range(self.number_of_features):
				action_include = self.action(self.ta_state[j,k,0])
				action_include_negated = self.action(self.ta_state[j,k,1])

				if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
					self.clause_output[j] = 0
					break
		print("clause output: {}".format(self.clause_output), file=open(self.logger, "a"))


	# Translate automata state to action 
	def action(self, state):
		if state <= self.number_of_states:
			return 0
		else:
			return 1

	# Get the state of a specific automaton, indexed by clause, feature, and automaton type (include/include negated).
	def get_state(self, clause, feature, automaton_type):
		return self.ta_state[clause,feature,automaton_type]

	# Sum up the votes for each output
	def sum_up_clause_votes(self):
		
		j = 0

		output_sum = 0
		for j in range(self.number_of_clauses):
			output_sum += self.clause_output[j]*self.clause_sign[j]
		
		if output_sum > (self.number_of_clauses * self.threshold/2):
			output_sum = self.threshold
		
		elif output_sum < 0:
			output_sum = 0
		print("Sum of clause votes: {}".format(output_sum), file=open(self.logger, "a"))
		return output_sum


	###########################################
	### Predict Target Output y for Input X ###
	###########################################

	def predict(self, X):
		output_sum = 0
		output_value = 0
		j = 0
		
		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X)

		###########################
		### Sum up Clause Votes ###
		###########################

        # Map the total clause outputs into a continuous value using max and min values of the target series
		output_sum = self.sum_up_clause_votes()
		output_value = ((output_sum * (self.max_target-self.min_target))/ self.threshold) + self.min_target
		print("Pred y: {}".format(output_value), file=open(self.logger, "a"))
		return output_value

	
	#######################################################
	### Evaluate the Trained Regression Tsetlin Machine ###
	#######################################################

	def evaluate(self, X, y, number_of_examples):
		j = 0
		l = 0
		errors = 0
		output_sum = 0

		Xi = np.zeros((self.number_of_features,), dtype=np.int32)


		for l in range(number_of_examples):
			###############################
			### Calculate Clause Output ###
			###############################

			for j in range(self.number_of_features):
				Xi[j] = X[l,j]

			errors += abs(self.predict(Xi) - y[l])

		return errors / number_of_examples

	#####################################################
	### Online Training of Regression Tsetlin Machine ###
	#####################################################

	# The Regression Tsetlin Machine can be trained incrementally, one training example at a time.
	# Use this method directly for online and incremental training.

	def update(self, X, y):
		i = 0
		j = 0
		action_include = 0
		action_include_negated = 0
		output_sum = 0
		output_value = 0
		print("FITTING:\nX:{}\ty:{}".format(X, y), file=open(self.logger, "a"))

		###############################
		### Calculate Clause Output ###
		###############################

		self.calculate_clause_output(X)

		###########################
		### Sum up Clause Votes ###
		###########################

		output_sum = self.sum_up_clause_votes()

        ##############################
		### Calculate Output Value ###
		##############################

		output_value = ((output_sum * (self.max_target-self.min_target))/ self.threshold) + self.min_target

		###########################################
		### Deciding the feedbck to each clause ###
		###########################################

		# Initialize feedback to clauses
		for j in range(self.number_of_clauses):
			self.feedback_to_clauses[j] = 0

        # Type I feedback if target is higher than the predicted value
		if y > output_value:
			for j in range(self.number_of_clauses):
				if 1.0*random.randint(0, RAND_MAX)/RAND_MAX < 1.0*(abs(y-output_value))/(self.max_target - self.min_target):
					self.feedback_to_clauses[j] += 1
					
        # Type II feedback if target is lower than the predicted value
		elif y < output_value:
			for j in range(self.number_of_clauses):
				if 1.0*random.randint(0, RAND_MAX)/RAND_MAX < 1.0*(abs(y-output_value))/(self.max_target - self.min_target):
					self.feedback_to_clauses[j] -= 1
		print("TA STATE BEFORE FEEDBACK:\n{}".format(self.ta_state), file=open(self.logger, "a"))
		print("Feedback type array: {}".format(self.feedback_to_clauses), file=open(self.logger, "a"))

		for j in range(self.number_of_clauses):
			if self.feedback_to_clauses[j] > 0:

				########################
				### Type I Feedback  ###
				########################
				
				if self.clause_output[j] == 0:		
					for k in range(self.number_of_features):	
						if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0/self.s:								
							if self.ta_state[j,k,0] > 1:
								self.ta_state[j,k,0] -= 1
													
						if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0/self.s:
							if self.ta_state[j,k,1] > 1:
								self.ta_state[j,k,1] -= 1

				if self.clause_output[j] == 1:					
					for k in range(self.number_of_features):
						if X[k] == 1:
							if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0*(self.s-1)/self.s:
								if self.ta_state[j,k,0] < self.number_of_states*2:
									self.ta_state[j,k,0] += 1

							if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,1] > 1:
									self.ta_state[j,k,1] -= 1

						elif X[k] == 0:
							if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0*(self.s-1)/self.s:
								if self.ta_state[j,k,1] < self.number_of_states*2:
									self.ta_state[j,k,1] += 1

							if 1.0*random.randint(0, RAND_MAX)/RAND_MAX <= 1.0/self.s:
								if self.ta_state[j,k,0] > 1:
									self.ta_state[j,k,0] -= 1
					
			elif self.feedback_to_clauses[j] < 0:
                
				#########################
				### Type II Feedback  ###
				#########################
				if self.clause_output[j] == 1:
					for k in range(self.number_of_features):
						action_include = self.action(self.ta_state[j,k,0])
						action_include_negated = self.action(self.ta_state[j,k,1])

						if X[k] == 0:
							if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
								self.ta_state[j,k,0] += 1
						elif X[k] == 1:
							if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
								self.ta_state[j,k,1] += 1
		print("TA STATES AFTER FEEDBACK:\n{}".format(self.ta_state), file=open(self.logger, "a"))


	#########################################################
	### Batch Mode Training of Regression Tsetlin Machine ###
	#########################################################

	def fit(self, X, y, number_of_examples, epochs=100):
		j = 0
		l = 0
		epoch = 0
		example_id = 0
		target_class = 0
		Xi = []
		random_index = []
		print("FITTING:\nX:{}\ty:{}".format(X, y), file=open(self.logger, "a"))
		Xi = np.zeros((self.number_of_features,), dtype=np.int32)
		random_index = np.arange(number_of_examples)

		for epoch in range(epochs):	
			np.random.shuffle(random_index)

			for l in range(number_of_examples):
				example_id = int(random_index[l])
				target_class = y[example_id]
            
				for j in range(self.number_of_features):
					Xi[j] = X[example_id,j]
				self.update(Xi, target_class)

		return


		