import custom
from custom import criterion
from custom.layers import *
from custom.config import config
from our_models import MusicTransformer
from data import Data
import utils
from midi_processor.processor import decode_midi, encode_midi

import datetime
import argparse

from tensorboardX import SummaryWriter


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    position_embedding=config.positional,
    relative_attention=config.relative,
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)



model_name = config.pickle_dir.split('/')[-1]

if config.positional:
    model_name = model_name + "_P"
    
if config.relative:
    model_name = model_name + "_R"

print(f"Loading model {model_name}")

mt.load_state_dict(torch.load(args.model_dir+'/model_'+model_name+'.pth'))

                              
mt.test()

print(config.condition_file)
if config.condition_file is not None:
    inputs = np.array([encode_midi(config.condition_file)[:128]])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs)
result = mt(inputs, config.length, gen_summary_writer)

for i in result:
    print(i)

condition_file_clean = (config.condition_file.split('/')[-1]).split('.')[0]
file_name = config.save_path + condition_file_clean

if config.positional:
    file_name = file_name + "_P"
    
if config.relative:
    file_name = file_name + "_R"

file_name = file_name + ".mid"

print(f"Saving file {file_name}")

decode_midi(result, file_path=file_name)

gen_summary_writer.close()
