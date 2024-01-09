import numpy as np
from task_tree import task_tree_node
from evaluation_tree import evaluation_tree_node
from student import student
from teacher import teacher
from task import task, setup
from typing import Callable

class replay_datum:

    def __init__(self, task_node:task_tree_node, next_task_node:task_tree_node, action_code:int, pi:np.ndarray, eval_root:evaluation_tree_node, task_index:int):

        self.task_node : task_tree_node = task_node
        self.next_task_node : task_tree_node = next_task_node
        self.action_code = action_code
        self.eval_tree_root : evaluation_tree_node = eval_root
        self.task_index = task_index
        self.pi = pi

class classroom:

    def __init__(self, task:task, setup:setup, teacher:teacher, student_template:student, n_students:int, max_steps:int, buffer_size:Callable):

        self.task = task
        self.setup = setup
        self.n_actions = task.n_actions

        self.teacher = teacher
        self.student :student = student_template
        self.n_students = n_students
        self.max_steps = max_steps

        self.input_buffer = None
        self.reward_buffer = None
        self.value_buffer = None

        self.buffer_size = buffer_size
        self.total_tasks = 0

    def run_training_batch(self, n_problems, epochs_per_episode):

        problems = self.teacher.generate_problems(n_problems)
        replay_record, proof_nodes = self.test_students(problems)

        self.total_tasks += n_problems
        inputs = []
        policies = []
        values = []
        reward_confidences = []
        rd:replay_datum

        for rd in replay_record:
            inputs.append(rd.task_node.state)
            policies.append(rd.pi)

            end_node:task_tree_node = proof_nodes[rd.task_index]
            values.append(self.task.reward_function(end_node.state)*self.student.reward_discount**(end_node.depth-rd.task_node.depth))

            #We only want to change the confidence for the estimated rewards for those 
            #actions that have actually been explored during the MCTS
            target_confidence = np.abs(rd.task_node.initial_reward-rd.pi)
            t = 1/(1+rd.eval_tree_root.n)
            reward_confidences.append(np.maximum(0.001,t*rd.task_node.reward_confidence+(1-t)*target_confidence))

        inputs = np.array(inputs)
        values = np.array(values)
        policies = np.array(policies)
        reward_confidences = np.array(reward_confidences)


        if self.input_buffer is None:
            self.input_buffer = inputs
            self.value_buffer = values
            self.reward_buffer = policies
            self.reward_confidence_buffer = reward_confidences
        else:
            target_buffer_size = self.buffer_size(self.total_tasks)
            end = target_buffer_size - inputs.shape[0]

            self.input_buffer = np.concatenate([inputs,self.input_buffer[:end]])
            self.value_buffer = np.concatenate([values, self.value_buffer[:end]])
            self.reward_buffer = np.concatenate([policies, self.reward_buffer[:end]])
            self.reward_confidence_buffer = np.concatenate([policies, self.reward_confidence_buffer[:end]])

        state_inputs = self.task.make_input(self.input_buffer)

        self.student.neural_network.fit_value(state_inputs,self.value_buffer,self.reward_buffer, self.reward_confidence_buffer, epochs_per_episode)

    def test_students(self, start_states):

        replay_record = []
        task_roots = []
        task_nodes = []

        for s in start_states:

            task_roots.append(task_tree_node(self.task, s,0))
        
        for s in range(self.n_students):

            task_nodes.extend(task_roots)

        for s in range(self.max_steps):

            unfinished_task_indices = [i for i,p in enumerate(task_nodes) if not p.completed]
            if len(unfinished_task_indices) == 0:
                break
            
            print(f"Before step {s+1}, {len(unfinished_task_indices)} out of {len(task_nodes)} remain open.")
            result = self.student.run_action_step([task_nodes[i] for i in unfinished_task_indices])

            for j, (action, pi, eval_root) in enumerate(result):

                k = unfinished_task_indices[j]
                old_node : task_tree_node = task_nodes[k]
                new_node : task_tree_node = old_node.children[action]
                task_nodes[k] = new_node

                datum = replay_datum(old_node,new_node,action, pi, eval_root, k)
                replay_record.append(datum)

        unfinished_task_indices = [i for i,p in enumerate(task_nodes) if not p.completed]
        print(f"After step {s+1}, {len(unfinished_task_indices)} out of {len(task_nodes)} remain open.")

        return replay_record, task_nodes