"""
A well-documented module to export the tensorflow trained graph as protobuf and
tflite.
Background on protobuf and tflite model format:
Protocol buffer format(frozen) and tflite format contains exactly the same graph
and same weights associated to that graph, the only difference is that tflite
format is understood by tflite interpreter while protobuf format is understood
by tensorflow.
In this module, three things are covered extensively:
1. How to process the tensorflow graph and recover nodes
2. How to determine which nodes to use for model exportation
3. Some technical explainations of variable_to_constant conversion and why it
is required.
How to use this model for exporting protobuf and tflite model from the
tensorflow trained checkpoint:
1. Identify the input and output node names along with the shapes.
set the export to False, read the terminal outputs
Identify the node name based on the input and output tensor shapes
2. Use tensorflow inbuilt APIs for freezing the model.
set the export to True, create a folder "generated_model"
3. use tensorflow inbuilt APIs for converting frozen model to protobuf/tflite.
set the infer to True and export to False.
Finally, check the inference result with the generated tflite model.
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import os

def print_node_names(checkpoint_dir="checkpoints", checkpoint_name="20191124-2200+l1+cyc+pr"):
    tf.reset_default_graph() # Clear any tf graph in the current thread
    checkpoint_file_address = os.path.join(checkpoint_dir, checkpoint_name)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_file_address)
    # There are two ways to construct the tensorlfow graph on the current thread
    # First, programatically call the create_model()
    # Second, import the graph defenition from checkpoint meta file
    # If first option is chosen, the graph defenition should match when
    # restoring the weight files from .data files (part of checkpoint).
    # Here, we go by second definition (import graph defenition from meta file)
    saver = tf.train.import_meta_graph(latest_ckpt + '.meta',
                   clear_devices = True) # Clear device associated to any tensor
    # By this time, all the nodes are added to the default tf graph, but this
    # graph contains all the nodes for training procedure as well.
    # We can print all the nodes in the current graph.
    sess = tf.get_default_session() # get track of the default session
    # get the default graph(already created from meta file) pointer.
    graph = tf.get_default_graph()
    # print all the node name assgined (by developer or automatically by tf)
    for op in graph.get_operations():
        print(op.name)
        inputs = op.inputs
        outputs = op.outputs
        print("----------------------------Inputs-----------------------------")
        for input in inputs:
            print(input.name, input.shape.as_list())
        print("----------------------------Outputs----------------------------")
        for output in outputs:
            print(output.name, output.shape.as_list())
        print("===============================================================")


def create_model(checkpoint_dir="checkpoints", checkpoint_name="20191124-2200+l1+cyc+pr",
       model_path = "generated_model", inputs = ["input"],outputs = ["output"]):
    tf.reset_default_graph()
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(checkpoint_dir,checkpoint_name))
    print(latest_ckpt)
    checkpoint_file_address = latest_ckpt
    saver = tf.train.import_meta_graph(checkpoint_file_address + '.meta',
                   clear_devices = True)
    sess = tf.Session()
    graph = tf.get_default_graph()
    # using the saver that we got by constructing the graph from meta file,
    # we restore the weight files on that constructed graph.
    saver.restore(sess, checkpoint_file_address)
    # In our current thread, we have lots of variables (values that can change)
    # To get a inference only graph, we can convert these variables to constant
    # this conversion is called freezing process
    simplified_graph_def = graph_util.convert_variables_to_constants(sess,
                                    graph.as_graph_def(), outputs[0].split(","))
    # this simplified graph is in proto format, and can be written as protobuf
    graph_io.write_graph(simplified_graph_def, model_path, 'generated_model.pb',
                                                                as_text = False)
    # now we will use tf converter to convert the protobuf model into tflite
    input_img_shape = {inputs[0] : [1, 256, 256, 3]}
    converter = tf.lite.TFLiteConverter.from_frozen_graph(model_path + \
           "/generated_model.pb", inputs, outputs, input_shapes=input_img_shape)
    tflite_model = converter.convert() # this is serialized, has to be written
    # as binary file.
    # Tflite takes care of optimizing (removing useless nodes) from the graph
    # We write the tflite_model to a tflite file
    open(model_path + "/generated_model.tflite", "wb").write(tflite_model)

def test_inference():
    pass

def main():
    # Set all the below variables as per requirement
    export = False # do you want to export the model to protobuf/tflite ?
    infer = False # do you want to check the inference result on expoted tflite?

    # Step 1: First get to know the nodes name (to identify the input/output)
    if not export:
        print_node_names()
    else:
        # By now, we have identified the input and output for the graph
        # Now, create the model based on this input and output tensor names
        inputs = ["Placeholder"] # Set this list appropriately
        outputs = ["DepthToSpace"] # Set this list appropriately
        create_model(inputs = inputs, outputs = outputs)

    if infer:
        test_inference()


if __name__ == "__main__":
    main()