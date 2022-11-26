/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
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
#include "board.h"
#include "action.h"
#include "weight.h"

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
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		std::string tuple4 = "65536,65536,65536,65536,65536,65536,65536,65536";
		std::string tuple6 = "16777216,16777216,16777216,16777216";
		meta["init"] = { tuple6 };
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

class td_agent : public weight_agent {
public:
	td_agent(const std::string& args = "") : weight_agent(args), 
		opcode({ 0, 1, 2, 3 }) {
			spaces[0] = { 12, 13, 14, 15 };
			spaces[1] = { 0, 4, 8, 12 };
			spaces[2] = { 0, 1, 2, 3};
			spaces[3] = { 3, 7, 11, 15 };
			spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
		}

public:
	struct step
	{
		int reward;
		board afterstate;
	}; // use to store afterstate and reward

public:
	virtual action take_action(const board& before) {
		int bestOP = -1;
		int bestReward = -1;
		float bestValue = -100000;
		board bestAfterstate;
		for(int op : opcode) {
			board afterstate = before; // use to store state after sliding
			int reward = afterstate.slide(op);
			if(reward == -1) continue;
			//float value = valueEstimate(afterstate);
			float value = expectationEstimate(afterstate);
			if(reward + value > bestReward + bestValue){
				bestReward = reward;
				bestValue = value;
				bestOP = op;
				bestAfterstate = afterstate;
			}
		}
		if(bestOP != -1){
			replayBuffer.push_back({bestReward, bestAfterstate});
		}
		return action::slide(bestOP);
	}
	int featureExtract(const board& after, int a, int b, int c, int d, int e, int f) const {
		//for 8*4-tuple
		//return after(a) * 16 * 16 * 16 + after(b) * 16 * 16 + after(c) * 16 + after(d);
		
		//for 4*6-tuple
		return after(a) * 16 * 16 * 16 * 16 * 16 + after(b) * 16 * 16 * 16 * 16 + after(c) * 16 * 16 * 16 + after(d) * 16 * 16 + after(e) * 16 + after(f);
	}
	float valueEstimate(const board& after) const {
		float value = 0;

		//for 8*4-tuple
		/*
		value += net[0][featureExtract(after, 0, 1, 2, 3)];
		value += net[1][featureExtract(after, 4, 5, 6, 7)];
		value += net[2][featureExtract(after, 8, 9, 10, 11)];
		value += net[3][featureExtract(after, 12, 13, 14, 15)];
		value += net[4][featureExtract(after, 0, 4, 8, 12)];
		value += net[5][featureExtract(after, 1, 5, 9, 13)];
		value += net[6][featureExtract(after, 2, 6, 10, 14)];
		value += net[7][featureExtract(after, 3, 7, 11, 15)];
		*/

		//for 4*6-tuple
		board state = after;
		for(int r = 0; r < 4; r++) {
			value += net[0][featureExtract(state, 0, 1, 2, 3, 4, 5)];
			value += net[1][featureExtract(state, 4, 5, 6, 7, 8, 9)];
			value += net[2][featureExtract(state, 0, 1, 2, 4, 5, 6)];
			value += net[3][featureExtract(state, 4, 5, 6, 8, 9, 10)];
			state.rotate_clockwise();
		}
		state.reflect_horizontal();
		for(int r = 0; r < 4; r++) {
			value += net[0][featureExtract(state, 0, 1, 2, 3, 4, 5)];
			value += net[1][featureExtract(state, 4, 5, 6, 7, 8, 9)];
			value += net[2][featureExtract(state, 0, 1, 2, 4, 5, 6)];
			value += net[3][featureExtract(state, 4, 5, 6, 8, 9, 10)];
			state.rotate_clockwise();
		}

		return value;
	}
	void valueAdjust(const board& after, float TDtarget) {
		float currentV = valueEstimate(after);
		float TDerror = TDtarget - currentV;
		float adjustment = alpha * TDerror;
		
		//These 8 feature weights are adjusted with the same TD error.
		/*
		net[0][featureExtract(after, 0, 1, 2, 3)] += adjustment;
		net[1][featureExtract(after, 4, 5, 6, 7)] += adjustment;
		net[2][featureExtract(after, 8, 9, 10, 11)] += adjustment;
		net[3][featureExtract(after, 12, 13, 14, 15)] += adjustment;
		net[4][featureExtract(after, 0, 4, 8, 12)] += adjustment;
		net[5][featureExtract(after, 1, 5, 9, 13)] += adjustment;
		net[6][featureExtract(after, 2, 6, 10, 14)] += adjustment;
		net[7][featureExtract(after, 3, 7, 11, 15)] += adjustment;
		*/

		//These is 4*6-tuple for eight isomorphic patterns.(32 features)
		board state = after;
		for(int r = 0; r < 4; r++) {
			net[0][featureExtract(state, 0, 1, 2, 3, 4, 5)] += adjustment/8;
			net[1][featureExtract(state, 4, 5, 6, 7, 8, 9)] += adjustment/8;
			net[2][featureExtract(state, 0, 1, 2, 4, 5, 6)] += adjustment/8;
			net[3][featureExtract(state, 4, 5, 6, 8, 9, 10)] += adjustment/8;
			state.rotate_clockwise();
		}
		state.reflect_horizontal();
		for(int r = 0; r < 4; r++) {
			net[0][featureExtract(state, 0, 1, 2, 3, 4, 5)] += adjustment/8;
			net[1][featureExtract(state, 4, 5, 6, 7, 8, 9)] += adjustment/8;
			net[2][featureExtract(state, 0, 1, 2, 4, 5, 6)] += adjustment/8;
			net[3][featureExtract(state, 4, 5, 6, 8, 9, 10)] += adjustment/8;
			state.rotate_clockwise();
		}
	}

	float expectationEstimate(const board& after) const {
		float expectation = 0.0;
		int emptySpace = 0;

		std::vector<int> space = spaces[after.last()];
		//std::shuffle(space.begin(), space.end(), engine);
		int bag[3], num = 0;
		for (board::cell t = 1; t <= 3; t++)
			for (size_t i = 0; i < after.bag(t); i++)
				bag[num++] = t;
		std::default_random_engine engine;
		std::shuffle(bag, bag + num, engine);
		board::cell tile = after.hint() ?: bag[--num];
		board::cell hint = bag[--num];

		for (int pos : space) {
			if (after(pos) != 0) continue;

			board b = board(after);
			b.place(pos, tile, hint); // place 1, 2, 3
			int bestReward = -1;
			float bestValue = -100000;
			for(int op : opcode){
				board afterstate = b;
				int reward = afterstate.slide(op);
				if(reward == -1) continue;
				float value = valueEstimate(afterstate);
				if(reward + value > bestReward + bestValue){
					bestReward = reward;
					bestValue = value;
					//bestOP = op;
					//bestAfterstate = afterstate;
				}
			}

			expectation += (bestReward + bestValue);
			emptySpace += 1;		
		}

		expectation = expectation / emptySpace;
		return expectation;

	}

	virtual void open_episode(const std::string & flag = ""){
		replayBuffer.clear();
	}
	virtual void close_episode(const std::string& flag = ""){
		if (replayBuffer.empty() || alpha == 0) return;
		//We have to update 0 to terminal afterstate so that it can converge.
		valueAdjust(replayBuffer[replayBuffer.size() - 1].afterstate, 0);
		//The backward method updates the afterstates from the end to the beginning.
		for(int t = replayBuffer.size() - 2; t >= 0; t--) {
			float TDtarget = replayBuffer[t+1].reward + valueEstimate(replayBuffer[t+1].afterstate);
			valueAdjust(replayBuffer[t].afterstate, TDtarget);
		}
	}

protected:
	std::vector<step> replayBuffer;
private:
	std::array<int, 4> opcode;
	std::vector<int> spaces[5];
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class reward_player : public agent {
public:
	reward_player(const std::string& args = "") : agent(args), 
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		//std::shuffle(opcode.begin(), opcode.end(), engine);
		int bestOP = -1;
		board::reward bestReward = -1;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward > bestReward){
				bestReward = reward;
				bestOP = op;
			}
		}
		if (bestOP != -1) {
			return action::slide(bestOP);
		}
		else {
			return action();
		}
	}

private:
	std::array<int, 4> opcode;
};

class twoSteps_player : public agent {
public:
	twoSteps_player(const std::string& args = "") : agent(args), 
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		//std::shuffle(opcode.begin(), opcode.end(), engine);
		int bestOP = -1;
		board::reward bestReward = -1;

		for (int op1 : opcode) {
			board next1 = board(before);
			board::reward reward1 = next1.slide(op1);
			for (int op2 : opcode) {
				board next2 = board(next1);
				board::reward reward2 = next2.slide(op2);
				if (reward1 + reward2 >= bestReward) {
					bestOP = op1;
					bestReward = reward1 + reward2;
				}
			}
		}

		if (bestOP != -1) {
			return  action::slide(bestOP);
		}
		else {
			return action();
		}
	}

private:
	std::array<int, 4> opcode;
};
