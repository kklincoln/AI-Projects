import random

from app.chat.redis import client

"""scores will be stored in redis dictionaries for LLM, retriever, and Memory components. Each respective dictionary will
    store the total of scores for the component selection; these are passed in dictionary keys shown in 
    pinecone.__init__, memories.__init__, and llms.__init__. There will also be a dictionary for count of scores, both will be used
     to calculate the averages, since redis doesn't allow for average computation out of box """

def random_component_by_score(component_type, component_map):
    """weighted randomness: random selection of component given component_type and component_map, considering the score """
    # make sure component type is "llm", "retriever", or "memory"
    if component_type not in ["llm", "retriever", "memory"]:
        raise ValueError("Invalid component_type")

    # from redis, get the hash containing the sum total scores for given component type
    values = client.hgetall(f"{component_type}_score_values")
    # from redis, get the hash containing the number of times each component has been scored
    counts = client.hgetall(f"{component_type}_score_counts")

    #get valid component names from component map (the keys from the dictionaries in init files) to pull out scores from redis
    names = component_map.keys()
    #loop over those valid names and use them to calculate the average score
    # add avg score to a dictionary
    avg_scores = {}
    for name in names:
        score = int(values.get(name, 1)) #everything in redis is stored as a string, convert to int.; return 1 if not found
        count = int(counts.get(name, 1))
        avg = score / count
        avg_scores[name] = max(avg, 0.1) #give everything at least a score of 0.1 in case the first vote is downvote
    # do a weighted random selection
    sum_scores = sum(avg_scores.values())
    random_val = random.uniform(0, sum_scores)
    cumulative = 0
    for name, score in avg_scores.items():
        # for each name, add the score into the cumulative; if the random_val <= cumulative, use that name.
        cumulative += score
        if random_val <= cumulative:
            return name



def score_conversation(
    conversation_id: str, score: int, llm: str, retriever: str, memory: str
) -> None:
    score = min(max(score, 0),1) #at most, our score can be 1, at min our score can be 0

    # llm_score_values refers to a hash in redis for llm_score_values; look at the key for the passed in llm; increment by score
    client.hincrby("llm_score_values", llm, score)
    client.hincrby("llm_score_counts", llm, 1)
    client.hincrby("retriever_score_values", retriever, score)
    client.hincrby("retriever_score_counts", retriever, 1)
    client.hincrby("memory_score_values", memory, score)
    client.hincrby("memory_score_counts", memory, 1)


def get_scores():
    # populates the scores in the web app from scores nav button
    aggregate = {"llm":{}, "retriever":{}, "memory":{}}

    for component_type in aggregate.keys():
        # from redis, get the hash containing the sum total scores for given component type
        values = client.hgetall(f"{component_type}_score_values")
        # from redis, get the hash containing the number of times each component has been scored
        counts = client.hgetall(f"{component_type}_score_counts")

        names = values.keys()
        for name in names:
            score = int(values.get(name, 1))
            count = int(counts.get(name, 1))
            avg = score / count
            aggregate[component_type][name] = [avg]   #update the aggregate dict on the component_type with [name] key:  avg

    return aggregate
