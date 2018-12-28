import subprocess, os

def export_inference_graph(model_dir, config_path):
  assert os.path.exists( config_path ), "File not found : {}".format(config_path)
  assert os.path.isdir( model_dir ), "Directory does not exists : {}".format(model_dir)

  # Find the last checkpoint
  ls = [ f for f in os.listdir( model_dir ) if f.startswith("model.ckpt-") ]
  ls.sort()
  best_checkpoint = '.'.join( ls[-1].split('.')[0:2] )
  
  args = ["python", "/avd/tensorflow/object_detection/export_inference_graph.py"]
  args.extend( ["--input_type", "image_tensor"] )
  args.extend( ["--pipeline_config_path", config_path] )
  args.extend( ["--trained_checkpoint_prefix", os.path.join(model_dir, best_checkpoint)] )
  args.extend( ["--output_directory", model_dir] )
  p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output, err = p.communicate()
  print(err.decode())
  print(output.decode())

#models = [f for f in os.listdir("/avd/output/") if not '.' in f]
#c_models = []

#for model in models:
#  export_inference_graph("/avd/output/" + model, model)
  
export_inference_graph("/avd/output/rfcn-2", "rfcn-2")