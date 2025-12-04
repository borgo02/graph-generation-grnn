import argparse
import yaml
import os

### program configuration
class Args():
    def __init__(self):
        # 1. Load config from file
        config = {}
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

        # 2. Define CLI args (defaults come from config or hardcoded fallback)
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
        parser.add_argument('--graph_type', type=str, default=config.get('graph_type', 'grid'), help='Graph type')
        parser.add_argument('--epochs', type=int, default=config.get('epochs', 3000), help='Number of epochs')
        parser.add_argument('--cuda', action='store_true', default=config.get('cuda', False), help='Use CUDA')
        parser.add_argument('--note', type=str, default=config.get('note', 'GraphRNN_RNN'), help='Note')
        parser.add_argument('--epochs_test', type=int, default=config.get('epochs_test', 100), help='Epochs test')
        parser.add_argument('--epochs_test_start', type=int, default=config.get('epochs_test_start', 100), help='Epochs test start')
        
        # Parse args
        args, unknown = parser.parse_known_args()
        
        # Reload config if a different config file was specified via CLI
        if args.config != 'config.yaml' and os.path.exists(args.config):
             with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                # Re-parse args to ensure CLI overrides new config defaults if necessary
                # (Simple approach: just trust CLI args > config file values for explicitly passed args)

        # 3. Set attributes (Priority: CLI > Config > Defaults)
        # We use the parsed args directly for things exposed in CLI
        self.graph_type = args.graph_type
        self.epochs = args.epochs
        self.cuda = args.cuda
        self.note = args.note
        self.epochs_test = args.epochs_test
        self.epochs_test_start = args.epochs_test_start
        self.clean_tensorboard = config.get('clean_tensorboard', False)

        # 4. Set other attributes from config (or defaults if missing)
        self.max_num_node = config.get('max_num_node', None)
        self.max_prev_node = config.get('max_prev_node', None)
        
        self.hidden_size_rnn = config.get('hidden_size_rnn', 128)
        self.hidden_size_rnn_output = config.get('hidden_size_rnn_output', 16)
        self.embedding_size_rnn = config.get('embedding_size_rnn', 64)
        self.embedding_size_rnn_output = config.get('embedding_size_rnn_output', 8)
        self.embedding_size_output = config.get('embedding_size_output', 64)
        self.num_layers = config.get('num_layers', 4)
        
        self.num_node_labels = config.get('num_node_labels', 12)
        self.label_embedding_size = config.get('label_embedding_size', 8)
        self.label_loss_weight = config.get('label_loss_weight', 1.0)
        
        # Adjust sizes based on graph type (legacy logic)
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
            
        self.hidden_size_rnn = int(self.hidden_size_rnn/self.parameter_shrink)
        self.embedding_size_rnn = int(self.embedding_size_rnn/self.parameter_shrink)
        self.embedding_size_output = int(self.embedding_size_output/self.parameter_shrink)

        self.batch_size = config.get('batch_size', 32)
        self.test_batch_size = config.get('test_batch_size', 32)
        self.test_total_size = config.get('test_total_size', 1000)
        
        self.num_workers = config.get('num_workers', 4)
        self.batch_ratio = config.get('batch_ratio', 32)
        self.epochs_log = config.get('epochs_log', 100)
        self.epochs_save = config.get('epochs_save', 100)
        
        self.lr = config.get('lr', 0.003)
        self.milestones = config.get('milestones', [400, 1000])
        self.lr_rate = config.get('lr_rate', 0.3)
        self.sample_time = config.get('sample_time', 2)
        
        self.dir_input = config.get('dir_input', "./")
        self.model_save_path = self.dir_input + 'model_save/'
        self.graph_save_path = self.dir_input + 'graphs/'
        self.figure_save_path = self.dir_input + 'figures/'
        self.timing_save_path = self.dir_input + 'timing/'
        self.figure_prediction_save_path = self.dir_input + 'figures_prediction/'
        self.nll_save_path = self.dir_input + 'nll/'
        
        self.load = config.get('load', False)
        self.load_epoch = config.get('load_epoch', 3000)
        self.save = config.get('save', True)
        
        self.generator_baseline = config.get('generator_baseline', 'BA')
        self.metric_baseline = config.get('metric_baseline', 'clustering')

        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline

