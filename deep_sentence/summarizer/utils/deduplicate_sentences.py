import numpy as np
import pandas as pd
from .sentences_similarity import sentences_similarity

from deep_sentence.logger import logger

def f1score(float1, float2):
    return (float1 + float2) / 2 * float1 * float2

# Every document is expected to have 3 sentence
def deduplicate_sentences(documents):
    document_output = []
    if len(documents) <= 1:
        return documents[0]

    if len(documents) == 2:
        document1, document2 = documents
        for i in range(3):
            sim = sentences_similarity(document1[i], document2[i])
            if sim >= 0.8 :
                logger.debug('[%d] %f > threshold! the longer sentence will be adopted.', i + 1, sim)
                if len(document1[i])> len(document2[i]):
                    document_output.append(document2[i])
                else:
                    document_output.append(document1[i])
            if sim < 0.8 :
                logger.debug('[%d] %f both sentences will be adopted.', i + 1, sim)
                document_output.append(document1[i])
                document_output.append(document2[i])
    else:
        document1, document2, document3 = documents[:3]
        for i in range(3):
            sim1n2 = sentences_similarity(document1[i], document2[i])
            sim2n3 = sentences_similarity(document2[i], document3[i])
            sim3n1 = sentences_similarity(document3[i], document1[i])
            sim = [f1score(sim1n2, sim3n1), f1score(sim1n2, sim2n3), f1score(sim3n1, sim2n3)]
            if np.argmax(sim) == 0:
                document_output.append(document1[i])
            elif np.argmax(sim) == 1:
                document_output.append(document2[i])
            elif np.argmax(sim) == 2:
                document_output.append(document3[i])

    return document_output


def main():
    pd.set_option('display.width', 1000)
    document1_lR = ['トランプ次期米大統領は１９日、就任式を翌日に控え、自宅があるニューヨークから首都ワシントン入りした。',
                    '出発に際し「旅が始まる」とツイッターに書き込んだトランプ氏は、ワシントンに到着後、戦没者が埋葬されている郊外のアーリントン国立墓地で献花し、歓迎イベントに出席。 ',
                    'コンサートの後のあいさつでは、「我々は真の変化を望んだ。（就任する）明日が楽しみだ」と強調し、「雇用を取り戻し、偉大な軍を作り上げ、国境を強化する。何十年にもわたって米国ができなかったことをする」などと語った。'
                    ]
    document2_lR = ['トランプ次期米大統領は１９日、任期が始まる２０日正午（日本時間２１日午前２時）に合わせた就任式のため首都ワシントンに入り、厳重な警護態勢の下で歓迎イベントに相次ぎ出席した。',
                    'ペンス氏は１９日の記者会見で「われわれは準備万端だ」と政権移行作業に胸を張った。',
                    'スパイサー次期大統領報道官は会見で、オバマ政権で国防副長官を務めたワーク氏ら主要な政府高官５０人に後任が決まるまで現職にとどまるよう求めたことを明らかにした。'
                   ]
    document3_lR = ['就任を控えたアメリカのトランプ次期大統領が、１９日、首都ワシントンに到着しました。',
                    '数万人が参加した歓迎式典にメラニア夫人や子どもたちと出席したトランプ氏は、予定されていなかった短いスピーチも行ないました。',
                    '「これはかつてなかった大きなうねりなんです。とてもとても特別な動きです。そして私たちは国を融和させるのです。国中のみんな、全ての人にとって、アメリカを偉大な国にするのです」（トランプ氏）。'
                   ]
    document_output = deduplicate_sentences([document1_lR, document2_lR, document3_lR])
    output = ''.join(str(x) for x in document_output)
    print(output)


if __name__ == '__main__':
    main()
