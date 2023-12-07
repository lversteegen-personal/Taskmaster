import numpy as np
from task_tree import task_tree_node
from evaluation_tree import evaluation_tree_node
from student import student
from teacher import teacher
from task import task, setup

class replay_datum:

    def __init__(self, task_node, next_task_node, action_code, eval_root, task_index):

        self.task_node : task_tree_node = task_node
        self.next_task_node : task_tree_node = next_task_node
        self.action_code = action_code
        self.eval_tree_root : evaluation_tree_node = eval_root
        self.task_index = task_index

class classroom:

    def __init__(self, task:task, setup:setup, teacher:teacher, student_template:student, n_students:int, max_steps:int):

        self.task = task
        self.setup = setup
        self.n_actions = task.n_actions

        self.teacher = teacher
        self.student :student = student_template
        self.n_students = n_students
        self.max_steps = max_steps

        self.replay_record = []

    def run_training_batch(self, n_problems):

        problems = self.teacher.generate_problems(n_problems)
        replay_record, proof_nodes = self.test_students(problems)

        inputs = []
        policies = []
        evals = []
        rd:replay_datum

        for rd in replay_record:
            inputs.append(rd.task_node.input)
            policies.append(rd.eval_tree_root.pi)
            if proof_nodes[rd.task_index].completed:
                evals.append(1.0)
            else:
                evals.append(0.0)

        inputs = np.array(inputs)
        evals = np.array(evals)
        policies = np.array(policies)

        self.student.neural_network.fit(inputs,evals,policies)

    def test_students(self, start_states):

        replay_record = []
        proof_roots = []
        proof_nodes = []

        for s in start_states:

            proof_roots.append(task_tree_node(self.task, s,0))
        
        for i in range(self.n_students):

            proof_nodes.extend(proof_roots)

        for i in range(self.max_steps):

            unfinished_task_indices = [i for i,p in enumerate(proof_nodes) if not p.completed]
            result = self.student.run_action_step([proof_nodes[i] for i in unfinished_task_indices])

            for j, (action, eval_root) in enumerate(result):

                k = unfinished_task_indices[j]
                old_node : task_tree_node = proof_nodes[k]
                new_node : task_tree_node = old_node.children[action]
                proof_nodes[k] = new_node

                datum = replay_datum(old_node,new_node,action,eval_root, k)
                replay_record.append(datum)

            print(f"Finished step {i}, {len(unfinished_task_indices)} out of {len(proof_nodes)} remain open.")

        return replay_record, proof_nodes