from copy import deepcopy

class ConfigObject:
    def __init__(self):
        pass

def get_embeddings_cost(in_c, vocab_size):
    """
    get cost of embeddings that are at the beginning of GPT / BERT / similar models

    This calculation looks only at the vocabulary embeddings. It ignores the sinusoidal embedding, because that only adds 1 channel.
    """
    return {
        'macs': 0.0,
        'params': in_c * vocab_size
    }

def get_pfc_cost(in_c, out_c, seq_len, batch):
    """ get cost of positionwise fully connected layer """
    return {
        'macs': in_c * out_c * seq_len * batch,
        'params': in_c * out_c
    }

def get_qk_cost(in_c, seq_len, batch):
    """ get cost of Q * (K^T) """
    return {
        'macs': in_c * seq_len * seq_len * batch,
        'params': 0.0
    }

def get_qkv_cost(in_c, seq_len, batch):
    """ get cost of (QK^T) * V """
    return {
        'macs': in_c * seq_len * seq_len * batch,
        'params': 0.0
    }

def get_gpt3_cost(conf):
    embeddings_cost = get_embeddings_cost(in_c=conf.hidden_size, vocab_size=conf.vocab_size)
    costs = dict()
    costs['q'] = get_pfc_cost(in_c=conf.hidden_size, out_c=conf.hidden_size, seq_len=conf.seq_len, batch=conf.batch)
    costs['k'] = costs['q']
    costs['v'] = costs['q']
    costs['qk'] = get_qk_cost(in_c=conf.hidden_size, seq_len=conf.seq_len, batch=conf.batch)
    costs['qkv'] = get_qkv_cost(in_c=conf.hidden_size, seq_len=conf.seq_len, batch=conf.batch)
    costs['ffn1'] = costs['q']
    costs['ffn2'] = get_pfc_cost(in_c=conf.hidden_size, out_c=conf.intermediate_size, seq_len=conf.seq_len, batch=conf.batch)
    costs['ffn3'] = get_pfc_cost(in_c=conf.intermediate_size, out_c=conf.hidden_size, seq_len=conf.seq_len, batch=conf.batch)

    per_module_macs = sum([v['macs'] for v in costs.values()])
    per_module_params = sum([v['params'] for v in costs.values()])

    total_macs = embeddings_cost['macs'] + conf.num_modules * per_module_macs
    total_params = embeddings_cost['params'] + conf.num_modules * per_module_params

    # note that I am ignoring the final classifier layer for now. It is usually cheap. Might add it layer.
    return {
        'macs': total_macs,
        'params': total_params
    }

def test_vs_gpt3_paper(conf):
    """ checking if #params matches what is reported in the GPT3 paper (175B params) """
    conf = deepcopy(conf)
    conf.seq_len = 2048 # Ani used  this; not sure why
    cost = get_gpt3_cost(conf)
    # TODO: assert that cost['params'] is within epsilon of 175B
    print(cost)
    print("")

def get_gpt3_autocomplete_cost(conf):
    """
    In gmail, as you type, an NLP system suggests the next few words.
    Here, I am envisioning that every time you type a word, a GPT3-based autocomplete system genrates the next `conf.autocomplete_len` words.
    (e.g. conf.autocomplete_len could be 5 words.)

    This calculation makes some simplifying assumptions...
    - it assumes that the final length of all your messages is `conf.seq_len`
    - it assumes that you don't go back and delete words, or copy-paste things. You just sit down, write 50 words, and that's it.
    - it assumes you add one word at a time to your sentence. This probably isn't quite the right assumption,
      because sometimes you will probably "accept" the next 5 words that GPT3 proposes and then you will keep writing from there.
    """
    _conf = deepcopy(conf)
    total_macs = 0.0
    for user_seq_len in range(1,conf.seq_len):
        for curr_rollout in range(0, conf.autocomplete_len):
            input_seq_len = user_seq_len + curr_rollout
            _conf.seq_len = input_seq_len
            cost = get_gpt3_cost(conf) # TODO: could memoize this if it proves to be expensive
            total_macs += cost['macs'] # TODO: if we have roundoff error, switch to tmacs and/or make sure it is a double and not a long int

    return {'macs': total_macs}

def get_worldwide_kwh_per_day():
    """
    sources: https://www.statista.com/statistics/280704/world-power-consumption/
        22,347 billion kwh in 2017 used globally
        = 22 trillion kwh
        = 22e12 kwh
    """
    kwh_per_year = 22e12
    kwh_per_day = kwh_per_year / 365
    return kwh_per_day

def get_kg_co2e_per_kwh():
    """
    kg of CO2e emitted per kwh

    sources:
    https://www.eia.gov/tools/faqs/faq.php?id=74&t=11
        has coal, natural gas, petroleum

    https://www.treehugger.com/how-much-co-does-one-solar-panel-create-4868753
        has solar
    """
    return {
        'coal': 2.21,
        'natural_gas': 0.92,
        'petroleum': 2.11,
        'solar': 0.05,
    }

def get_impact_of_macs(macs, tmac_per_sec, gpu_watts, num_days):
    """
    macs = macs required to process in num_days worth of messages
    tmac_per_sec = achievable throughput on 1 gpu
    gpu_watts = typical power draw for 1 gpu when at full load
    num_days = macs used in `num_days` worth of time
    """
    assert num_days==1, "num_days must be 1 for now. might implement a more general version later."
    macs_needed_per_sec = macs / (24 * 60 * 60)
    tmacs_needed_per_sec = macs_needed_per_sec / 1e12
    num_gpus_needed = tmacs_needed_per_sec / tmac_per_sec
    gpu_kwatts = gpu_watts / 1000
    total_kwatts = num_gpus_needed * gpu_kwatts
    kwh = total_kwatts * 24 # 24 hrs per day


    return {
        'tmacs_needed_per_sec': tmacs_needed_per_sec,
        'num_gpus_needed': num_gpus_needed,
        'kwh': kwh,
        'gwh': kwh / 1e6,
        'pct_of_global_energy': 100 * kwh / get_worldwide_kwh_per_day(),
        'daily_kg_co2e_if_coal': kwh * get_kg_co2e_per_kwh()['coal'],
        'daily_gigatons_co2e_if_coal': kwh * (get_kg_co2e_per_kwh()['coal'] / 1e12), # metric ton = 1e3 kg; gigaton = 1e9 metric tons, so divide by 1e12 to convert kg to gigatons. (US emits ~6 gigatons per year)
        # TODO: gigatons number seems low.
        'annual_gigatons_co2e_if_coal': 365 * kwh * (get_kg_co2e_per_kwh()['coal'] / 1e12), # metric ton = 1e3 kg; gigaton = 1e9 metric tons, so divide by 1e12 to convert kg to gigatons. (US emits ~6 gigatons per year)

    }


if __name__ == "__main__":
    conf = ConfigObject()
    conf.num_modules = 96
    conf.hidden_size = 12288
    conf.intermediate_size = conf.hidden_size*4
    conf.embedding_size = conf.hidden_size
    conf.vocab_size = 30000
    conf.batch = 1.0 # make calculations in float -- more resilient to roundoff
    conf.seq_len = 50 # TODO: think more about this. I remember reading that the avg facebook post is around 50 or 70 words long
    conf.autocomplete_len = 5 # for autocomplete use-case, predict the next 5 words
    conf.messages_per_day = 3e11 # three hundred billion
    """
    note that we are ignoring the number of heads for now.
    The # of heads affects the number of activations and the arithmetic intensity, but it does not affect params or macs.
    """

    #### sanity-check
    test_vs_gpt3_paper(conf)


    #### classification use-case
    gpt3_cost = get_gpt3_cost(conf)
    tot_gpt3_cost = gpt3_cost['macs'] * conf.messages_per_day
    impact = get_impact_of_macs(macs=tot_gpt3_cost, tmac_per_sec=30, gpu_watts=250, num_days=1)
    print(f"imagine you trained a GPT3-sized model to do text classification (like BERT). Here is what it would cost to run on every message generated by humanity each day: {impact}")
    print("")

    #### autocomplete use-case
    gpt3_cost = get_gpt3_autocomplete_cost(conf)
    tot_gpt3_cost = gpt3_cost['macs'] * conf.messages_per_day
    impact = get_impact_of_macs(macs=tot_gpt3_cost, tmac_per_sec=30, gpu_watts=250, num_days=1)
    print(f"Here is what it would cost to use GPT3 as an autocomplete system for every message generated by humanity each day: {impact}")
    print("")
