/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <deque>
#include <omp.h>
#include "board.h"
#include "action.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

public:
	class node{
	public:
		board state;		
		int wins; 
		int total; 
		int pos;		
		std::vector<node*> children;
		std::unordered_set<int> children_pos;
		node* parent;

		node(const board& b): state(b), wins(0), total(0), pos(-1), parent(nullptr) {}
 
		float calculateWinRate(){
			return (total == 0) ? 0.0 : ((float) wins / total);
		}

		float calculateUCB(){
			return (parent->total == 0 || total == 0) ? calculateWinRate() : (calculateWinRate() + 0.5 * std::sqrt(std::log(parent->total) / total));
		} 		
	};

public:
	std::vector<node*> takeSelection(node* root){
		std::vector<node*> vec;
		node* current = root; // root
		vec.push_back(current);

		int cnt = 0;
		for (const action::place& move : space) {
			board b = current->state;
			if (move.apply(b) == board::legal)
				cnt++;
		}
		bool leaf = (cnt == 0 || current->children.size() != cnt);
		
		while(!leaf){
			float value = -1000000.0;
			node* nextNode;

			for(auto &childNode : current->children){
				float tmp = childNode->calculateUCB();
				if(tmp > value){
					value = tmp;
					nextNode = childNode; 
				}
			}

			vec.push_back(nextNode);
			current = nextNode;

			// check whether is fully expanded
			cnt = 0;
			for (const action::place& move : space) {
				board b = current->state;
				if (move.apply(b) == board::legal)
					cnt++;
			}
			leaf = (cnt == 0 || current->children.size() != cnt);
		}

		return vec;
	}

	node* takeExpansion(node* Node){
		board b;
		bool success = false;
		int pos = -1;
		std::vector<int> vec;
		for(int i = 0; i < 81; ++i){
			vec.push_back(i);
		}
		std::random_shuffle(vec.begin(), vec.end());

		for(int i = 0; i < vec.size(); i++){
			b = Node->state;
			if(b.place(vec[i]) == board::legal){
				if (!Node->children_pos.count(vec[i])){
					pos = vec[i];
					success = true;
					break;
				}
			}
		}

		if(success){
			node* newNode = new node(b);
			newNode->pos = pos;
			Node->children.push_back(newNode);
			Node->children_pos.insert(pos);
			newNode->parent = Node;
			return newNode;
		}else{
			return Node;
		}
	}

	int takeSimulation(board& state){
		//board b = Node->state;
		board b = state;
		std::deque<int> dq;
		for(int i = 0; i < 81; ++i){
			dq.push_back(i);
		}
		std::random_shuffle(dq.begin(), dq.end());

		int remaining = dq.size();
		while(remaining != 0){
			int i = dq.front();
			dq.pop_front();
			if(b.place(i) == board::legal){
				remaining = dq.size();
			}else{
				dq.push_back(i);
				remaining--;
			}
		}

		if(b.info().who_take_turns == board::black){
			//return board::white;
			return 1;
		}else{
			//return board::black;
			return -1;
		}
	}

	void takeBackpropagation(std::vector<node*>& path, unsigned winner, int games, int win_count){
		for(int i = 0; i < path.size(); i++){
			path[i]->total = path[i]->total + games;
			if(winner != (path[i]->state.info()).who_take_turns){
				path[i]->wins = path[i]->wins + win_count;
			}else{
				path[i]->wins = path[i]->wins - win_count;
			}
			
		}
	}

	int MCTS(node* root, int count){
		for (int i = 0; i < count; i++){
			// 1. Selection
			std::vector<node*> path = takeSelection(root);
			// 2. Expansion
			node* expandNode = takeExpansion(path.back());
			if (expandNode != path.back()){  // check whether is terminal node
				path.push_back(expandNode);
			}
			// 3. Simualtion
			int sum = takeSimulation(path.back()->state);
			
			// leaf parallel
			/*int THREAD_NUM = omp_get_num_threads();
			std::vector<int> winner(THREAD_NUM, 0);
			std::vector<board> chosenNode(THREAD_NUM, path.back()->state);
			#pragma omp parallel
			{
				int id = omp_get_thread_num();
				winner[id] = takeSimulation(chosenNode[id]);
			}
			int sum = 0;
			#pragma omp parallel for reduction(+:sum)
			for (int id = 0; id < THREAD_NUM; id++){
					sum += winner[id];
			}*/
			
			unsigned Winners;
			if (sum > 0){
				Winners = board::white;
			}else{
				Winners = board::black;
			}

			// 4. Backpropagation
			int THREAD_NUM = 1;
			takeBackpropagation(path, Winners, THREAD_NUM, abs(sum));
		}

		// takeAction
		if(root->children.size() == 0){
			return -1;
		}

		int value = -1000000;
		int res = -1;
		for(auto &childNode : root->children){
			int tmp = childNode->total;
			if(tmp > value){
				value = tmp;
				res = childNode->pos; 
			}
		} 

		return res;
	}

	void freeTree(node* root){
		if(root->children.size() == 0){
			delete root;
		}else{
			for(auto &childNode : root->children){
				if(childNode != nullptr){
					freeTree(childNode);
				}
			}

			delete root;
		}		
	}

public:
	virtual action take_action(const board& state) {
		if (meta.find("mcts") != meta.end()){
			int N = 5000; // initial value, best value: 15000
			if (meta.find("count") != meta.end())
				N = std::stoi(meta["count"]);
			
			// MCTS
			// Root Parallelization
			int THREAD_NUM = omp_get_num_threads();
			std::vector<int> res_parallel(THREAD_NUM, 0);
			#pragma omp parallel
			{
				int id = omp_get_thread_num();
				node* root = new node(state);
				int res = MCTS(root, N);
				freeTree(root);
				res_parallel[id] = res;
			}
			std::vector<int> majorityVote(81, 0);
			int max_count = 0;
			int result = -1;
			for(int i = 0; i < res_parallel.size(); i++){
				int tmp = res_parallel[i];
				if(tmp != -1){
					majorityVote[tmp]++;
					if (majorityVote[tmp] > max_count){
						max_count = majorityVote[tmp];
						result = tmp;
					}
				}
			}
			if(result != -1){
				return action::place(result, state.info().who_take_turns);
			}else{
				return action();
			}

			/*node* root = new node(state);
			int result = MCTS(root, N);
			freeTree(root);
			if(result != -1){
				return action::place(result, state.info().who_take_turns);
			}else{
				return action();
			}*/
		}
		else{  
			// original random agent code
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : space) {
				board after = state;
				if (move.apply(after) == board::legal)
					return move;
			}
			return action();
		}
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};
