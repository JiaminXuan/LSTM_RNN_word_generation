require 'cunn';
require('nngraph')
stringx = require('pl.stringx')
require 'io'
require('base')
function transfer_data(x)
  return x:cuda()
end
-- parameters used for the pretrained model
params = {batch_size=50,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1000,
                dropout=0.65,
                init_weight=0.04,
                lr=0.5,
                vocab_size=50,
                max_epoch=4,
                max_max_epoch=50,
                max_grad_norm=10}


--loading model and everything else
function loading()
  model = torch.load('/scratch/jx624/model.rnn')
  --loading vocabulary map
  vocab_map= torch.load('vop')
  -- reset state
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
  --initialization 
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  -- loading predicting word
  x = transfer_data(torch.ones(params.batch_size))
  -- doesnt matter what y is
  y = transfer_data(torch.ones(params.batch_size))
  io.write("\nOK GO\n")
  io.flush()
end
function main()
  loading()
  while true do
    line = io.read("*line")
    line = stringx.split(line)
    -- if a space or blank is entered, an underscore is returned as predicting character
    if next(line) == nil then word_to_predict = '_' else word_to_predict = line[1] end
    -- get the index in the vocab map of the character
    idx = vocab_map[word_to_predict]
    -- fill x with the same character
    x:fill(idx)
    perp_c, model.s[1], pred_c = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    -- get prediction back to CPU
    next_word = pred_c[1]:clone():float()
    -- print out every log probability in pred tensor
    for i=1,next_word:size(1) do
        io.write(next_word[i],' ')
        io.flush()
    end
  g_replace_table(model.s[0], model.s[1])
  io.write('\n')
  io.flush()
  end
end
main()