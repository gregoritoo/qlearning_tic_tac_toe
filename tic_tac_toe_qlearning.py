import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Game():

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.start_game()
        self.winner="begin"


    def start_game(self):
        self.grille = {'6': ' ' , '7': ' ' , '8': ' ' ,
            '3': ' ' , '4': ' ' , '5': ' ' ,
            '0': ' ' , '1': ' ' , '2': ' ' }
        self.end=False

    def play_x(self,choice):
        self.grille[str(choice)]="x"
 

    def play_o(self,choice):
        self.grille[str(choice)]="o"

    def get_grille(self):
        return self.grille

    def game_turn_x(self,choice):
        self.print_grille()
        self.end=False
        print('x turn')
        self.play_x(choice)
        self.analyse_state()

    def game_turn_o(self,choice) :   
        self.print_grille()
        print("o turn ")
        self.play_o(choice)
        self.analyse_state()
    
    def game_turn_o_solo(self):
        self.print_grille()
        print("o turn ")
        choice=int(input())
        while self.grille[str(choice)] != ' ':
            print("o turn ")
            choice=int(input())
        self.play_o(choice)
        self.print_grille()
        self.analyse_state()

    
    def analyse_state(self):
        if  self.grille['6'] ==  self.grille['7'] ==  self.grille['8'] != ' ': # across the top
            self.end=True    
            self.winner=self.grille['6']        
        elif  self.grille['3'] ==  self.grille['4'] ==  self.grille['5'] != ' ': # across the middle
            self.end=True 
            self.winner=self.grille['3'] 
        elif  self.grille['0'] ==  self.grille['1'] ==  self.grille['2'] != ' ': # across the bottom
            self.end=True 
            self.winner=self.grille['0']   
        elif  self.grille['0'] ==  self.grille['3'] ==  self.grille['6'] != ' ': # down the left side
            self.end=True 
            self.winner=self.grille['0']
        elif  self.grille['1'] ==  self.grille['4'] ==  self.grille['7'] != ' ': # down the middle
            self.end=True 
            self.winner=self.grille['1']
        elif  self.grille['2'] ==  self.grille['5'] ==  self.grille['8'] != ' ': # down the right side
            self.end=True 
            self.winner=self.grille['2']
        elif  self.grille['6'] ==  self.grille['4'] ==  self.grille['2'] != ' ': # diagonal
            self.end=True
            self.winner=self.grille['6'] 
        elif  self.grille['0'] ==  self.grille['4'] ==  self.grille['8'] != ' ': # diagonal
            self.end=True
            self.winner=self.grille['0'] 
        elif self.grille['0'] != ' ' and self.grille['1'] != ' ' and self.grille['2'] != ' ' and self.grille['3'] != ' ' and self.grille['4'] != ' ' and self.grille['5'] != ' ' and self.grille['6'] != ' ' and self.grille['7'] != ' ' and self.grille['8'] != ' ' :
            self.end=True
            self.winner="none"
  
    
    def print_grille(self):
        print(self.grille['6'] + '|' + self.grille['7'] + '|' + self.grille['8'])
        print('-+-+-')
        print(self.grille['3'] + '|' + self.grille['4'] + '|' + self.grille['5'])
        print('-+-+-')
        print(self.grille['0'] + '|' + self.grille['1'] + '|' + self.grille['2'])
        print("                             ")



class q_player():

    def __init__(self,actions_n):
        self.states_n = 0
        self.nb_state= 0
        self.actions_n = actions_n
        self.Q = np.zeros((1,9))
        self.pos= 1
        self.played_memory=np.array([])

        # Set learning parameters
        self.lr = .85
        self.y = .90
        self.cumul_reward = 0
        self.actions = []
        self.state = {}
        self.wrong_action_p=0.15

    def get_state_from_grille(self,grille,turn=True):
    	if grille in self.state.values() :
    		self.pos=self.state.keys()[self.state.values().index(grille)]
    		print("the state already exist")
    	elif grille not in self.state.values() and turn ==True :
    		self.nb_state =self.nb_state + 1
    		self.state.update({ self.nb_state  : grille.copy()})
    		self.pos=self.nb_state
    		new_row=np.zeros((1,self.actions_n))
    		self.Q=np.vstack([self.Q, new_row])
    	return self.pos


    def choose_move(self,i,grille):
        self.pos=self.get_state_from_grille(grille)
        Q2 = self.Q[int(self.pos)][:] + np.random.randn(1, self.actions_n)*(1. / (i +1))
        a = np.argmax(Q2)
        choice = random.random()
        if choice < self.wrong_action_p:
            a = (a ) % 9
            a=int(a)
        j=0
        while grille[str(a)] != ' ' :
            print("in while")
            if j < 10 :
                Q2 = self.Q[int(self.pos)][:] + np.random.randn(1, self.actions_n)*(1. / (i +1))
                a = np.argmax(Q2)
                choice = random.random()
                if choice < self.wrong_action_p:
                    a = (a + 1) % 9
                elif choice < 2 * self.wrong_action_p:
                    a = (a - 1) % 9
                    a=int(a)
                j=j+1
            else :
                try :
                    empty=self.find_empty_case(grille)
                    a=random.choice(empty)
                    a=a%9
                except Exception as e :
                    print(e)
                    break
        return int(a)

    def find_empty_case(self,grille):
        empty=[]
        print(grille)
        for i in range(9):
            if grille[str(i)]== ' ':
                print(i)
                empty=empty+[i]
        return empty


    def choose_move_trained(self,grille):
        self.pos=self.get_state_from_grille(grille)
        Q2 = self.Q[int(self.pos),:] + np.random.randn(1, self.actions_n)*1
        a = np.argmax(Q2)
        choice = random.random()
        j=0
        while grille[str(a)] != ' ' :
            if j < 10 :
                if choice < self.wrong_action_p:
                    a = (a + 1) % 9
                elif choice < 2 * self.wrong_action_p:
                    a = (a - 1) % 9
                j=j+1
            else :
                try :
                    empty=self.find_empty_case(grille)
                    print(empty)
                    a=random.choice(empty)
                    a=a%9
                except Exception as e :
                    print(e)
                    break
        return int(a)
    
    def update_Q(self,reward,a,grille):
    	self.new_pos=self.get_state_from_grille(grille,turn=False)
        self.Q[self.pos][a] = (1-lr)*self.Q[self.pos][a] + lr*(reward + y * np.max(self.Q[self.new_pos][:]) - self.Q[self.pos][a])
        self.played_memory=np.append(self.played_memory,[self.pos,a])
        self.cumul_reward =  self.cumul_reward + reward
        print(self.Q)
        print(self.played_memory)




if __name__ == "__main__":
    train_game=20000
    lr = 0.75
    y = 0.90
    q1cumul_reward_list = []
    q1actions_list = []
    q1states_list = []
    q2cumul_reward_list = []
    q2actions_list = []
    q2states_list = []
    winner1=[]
    with open('player.pkl', 'rb') as model:
        q_player = pickle.load(model)
    q_player1=q_player
    q_player2=q_player(9)
    for i in range(train_game):
        game=Game(3,3)
        grille=game.get_grille()
        q1actions = []
        q1states = [grille]
        q1cumul_reward = 0
        q2actions = []
        q2states = [grille]
        q2cumul_reward = 0
        while game.end == False :
            grille=game.get_grille()
            choice_1=q_player1.choose_move(i,grille)
            game.game_turn_x(choice_1)
            if game.end == True and game.winner == "x" :
                reward=11
            elif  game.end == True and game.winner == "o" :
                reward=-11
            elif  game.end == True :
                reward=2
            else :
                reward=-1
            if game.winner=="x":
                winner1=winner1+[1.0/train_game]
            else :
                winner1=winner1+[0.0/train_game]
            grille=game.get_grille()
            q_player1.update_Q(reward,choice_1,grille)
            q1cumul_reward += reward
            if game.end == False :
                grille=game.get_grille()
                choice_2=q_player2.choose_move(i,grille)
                game.game_turn_o(choice_2)
                if game.end == True and game.winner == "o" :
                    reward=11
                elif  game.end == True and game.winner == "X":
                    reward=-11
                elif  game.end == True  :
                    reward=2
                else :
                    reward= -1
                grille=game.get_grille()
                q_player2.update_Q(reward,choice_2,grille)
                q2cumul_reward += reward
        q_player1.pos=0
        q_player2.pos=0
        q_player1.played_memory=np.array([])
        q_player2.played_memory=np.array([])
        game.print_grille()
        q1cumul_reward_list.append(q1cumul_reward)
        q2cumul_reward_list.append(q2cumul_reward)
    winner=np.cumsum(winner1)
    winner2=[1.0-x for x in winner]
    plt.plot(winner,label="player 1")
    plt.plot(winner2,label="player2")
    plt.legend()
    plt.title("Percentage of games won")
    plt.show()
    plt.plot(np.cumsum(q1cumul_reward_list)/train_game,label="player 1")
    plt.plot(np.cumsum(q2cumul_reward_list)/train_game,label="player 2")
    plt.title("Average reward ")
    plt.legend()
    plt.show()
    if  q1cumul_reward_list > q2cumul_reward_list :
        q_player=q_player1
    else :
        q_player=q_player2
    with open('player.pkl', 'wb') as output:
        pickle.dump(q_player1, output)
    game2=Game(3,3)
    grille=game2.get_grille()
    while game2.end == False :
        grille=game2.get_grille()
        choice_1=q_player.choose_move_trained(grille)
        game2.game_turn_x(choice_1)
        reward=-1
        s1=game2.get_grille()
        q_player.update_Q(reward,choice_1,grille)
        game2.print_grille()
        if game2.end == False :
            game2.game_turn_o_solo()
    game2.print_grille()
    print("And the winner is ",game2.winner)
