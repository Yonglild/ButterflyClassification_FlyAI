import matplotlib.pyplot as plt
import numpy as np

def plot_bar(labels_nums):
    plt.bar([x[0] for x in labels_nums], [x[1] for x in labels_nums])
    plt.show()
    # plt.savefig('bar.png')


"""训练集分布"""
train_labels_nums = [('Gonepteryx_rhamni', 702), ('Catopsilia_pomona', 342), ('Eurema_hecabe', 338), ('Neptis_hylas', 330), ('Hebomoia_glaucippe', 310), ('Apatura_ilia', 292), ('Apatura_iris', 256), ('Eurema_blanda', 250), ('Junonia_orithya', 229), ('Junonia_iphita', 228), ('Delias_pasithoe', 219), ('Junonia_atlites', 217), ('Junonia_almana', 209), ('Symbrenthia_lilaea', 201), ('Cethosia_biblis', 201), ('Danaus_chrysippus', 194), ('Clossiana_euphrosyne', 188), ('Kallima_inachus', 180), ('Parnara_guttata', 177), ('Vanessa_cardui', 175), ('Colias_palaeno', 175), ('Junonia_hierta', 173), ('Zemeros_flegyas', 172), ('Parnassius_phoebus', 168), ('Acraea_issoria', 167), ('Pazala_eurous', 167), ('Iphiclides_podalirius', 167), ('Parantica_aglea', 166), ('Ideopsis_similis', 166), ('Colias_erate', 164), ('Danaus_plexippus', 157), ('Papilio_maackii', 154), ('Cethosia_cyane', 148), ('Tirumala_septentrionis', 144), ('Pathysa_antiphates', 142), ('Atrophaneura_horishanus', 142), ('Pachliopta_aristolochiae', 138), ('Byasa_alcinous', 136), ('Troides_magellanus', 133), ('Byasa_polyeuctes', 130), ('Colias_hyale', 129), ('Hypolimnas_bolina', 127), ('Pieris_canidia', 126), ('Sericinus_montelus', 126), ('Hestina_assimilis', 125), ('Kaniska_canace', 125), ('Polyura_athamas', 125), ('Faunis_eumeus', 125), ('Libythea_lepita', 125), ('Abisara_echerius', 124), ('Pieris_melete', 121), ('Cyrestis_thyodamas', 119), ('Elymnias_hypermnestra', 119), ('Mycalesis_perseus', 118), ('Udaspes_folus', 117), ('Mycalesis_gotama', 117), ('Papilio_hermosanus', 116), ('Melanitis_leda', 113), ('Iambrix_salsala', 110), ('Papilio_hoppo', 109), ('Melanitis_phedima', 108), ('Loxura_atymnus', 108), ('Notocrypta_curvifascia', 106), ('Polygonia_caureum', 105), ('Curetis_acuta', 105), ('Hasora_badra', 105), ('Pieris_rapae', 104), ('Parantica_sita', 103), ('Lethe_confusa', 103), ('Parnassius_apollo', 102), ('Pseudocoladenia_dan', 102), ('Tirumala_limniace', 101), ('Danaus_genutia', 98), ('Athyma_perius', 98), ('Odontoptilum_angulatum', 97), ('Stichophthalma_howqua', 97), ('Lethe_chandica', 96), ('Pieris_napi', 95), ('Cepora_nerissa', 94), ('Vanessa_indica', 93), ('Acraea_terpsicore', 92), ('Iraota_timoleon', 92), ('Leptidea_sinapis', 92), ('Eurema_laeta', 91), ('Clossiana_dia', 91), ('Daimio_tethys', 91), ('Troides_helena', 91), ('Badamia_exclamationis', 89), ('Papilio_machaon', 88), ('Ypthima_baldus', 87), ('Parantica_melaneus', 87), ('Papilio_prexaspes', 86), ('Taraka_hamada', 86), ('Rohana_parisatis', 86), ('Papilio_bianor', 85), ('Atrophaneura_varuna', 85), ('Troides_aeacus', 84), ('Lampides_boeticus', 84), ('Gonepteryx_amintha', 83), ('Papilio_dialis', 83), ('Horaga_onyx', 83), ('Burara_gomata', 79), ('Papilio_memnon', 79), ('Erionota_torus', 78), ('Athyma_ranga', 78), ('Euploea_tulliolus', 76), ('Isoteinon_lamprospilus', 76), ('Eurema_mandarina', 76), ('Euploea_core', 75), ('Artipe_eryx', 75), ('Dodona_eugenes', 75), ('Papilio_arcturus', 75), ('Stibochiona_nicea', 74), ('Luehdorfia_chinensis', 74), ('Eurema_andersoni', 74), ('Neope_pulaha', 72), ('Euploea_midamus', 72), ('Sasakia_charonda', 72), ('Charaxes_bernardus', 71), ('Ariadne_ariadne', 71), ('Clossiana_titania', 71), ('Papilio_xuthus', 68), ('Issoria_lathonia', 68), ('Colias_fieldii', 68), ('Mycalesis_intermedia', 68), ('Polyura_narcaea', 68), ('Astictopterus_jama', 68), ('Pazala_mullah', 67), ('Limenitis_sulpitia', 67), ('Amblopala_avidiena', 67), ('Timelaea_albescens', 67), ('Ampittia_virgata', 65), ('Polyura_eudamippus', 64), ('Abraximorpha_davidii', 64), ('Neptis_miah', 64), ('Calinaga_buddha', 63), ('Zizeeria_maha', 63), ('Miletus_chinensis', 63), ('Parnassius_nomion', 63), ('Graphium_sarpedon', 62), ('Libythea_myrrha', 61), ('Papilio_alcmenor', 61), ('Papilio_polytes', 61), ('Hasora_anura', 61), ('Papilio_paris', 60), ('Delias_descombesi', 60), ('Gandaca_harina', 58), ('Lamproptera_curius', 57), ('Clossiana_freija', 57), ('Catopsilia_scylla', 57), ('SpinDasis_syama', 57), ('Ariadne_merione', 57), ('Byasa_dasarada', 56), ('Delias_acalis', 56), ('Ancistroides_nigrita', 56), ('Proclossiana_eunomia', 56), ('Lamproptera_meges', 56), ('Teinopalpus_imperialis', 55), ('Lethe_syrcis', 55), ('Rapala_nissa', 55), ('Losaria_coon', 55), ('Tagiades_menaka', 54), ('Graphium_agamemnon', 52), ('Papilio_protenor', 52), ('Hasora_vitta', 51), ('Delias_belladonna', 51), ('Euploea_sylvester', 50), ('Teinopalpus_aureus', 50), ('Catopsilia_pyranthe', 50), ('Aemona_amathusia', 49), ('Celaenorrhinus_maculosus', 49), ('Doleschallia_bisaltide', 49), ('Bhutanitis_lidderdalii', 48), ('Meandrusa_payeni', 48), ('Penthema_darlisa', 47), ('Damora_sagana', 47), ('Ideopsis_vulgaris', 47), ('Eurema_brigitta', 46), ('Tongeia_potanini', 45), ('Horaga_albimacula', 45), ('Papilio_nephelus', 44), ('Graphium_cloanthus', 43), ('Jamides_bochus', 42), ('Meandrusa_sciron', 42), ('Mandarinia_regalis', 42), ('Leptidea_amurensis', 41), ('Idea_leuconoe', 41), ('Penthema_formosanum', 41), ('Ussuriana_michaelis', 40), ('Mahathala_ameria', 39), ('Sephisa_chandra', 38), ('Leptidea_morsei', 35), ('Papilio_Krishna', 34), ('Arhopala_rama', 34), ('Allotinus_drumila', 33), ('Ypthima_praenubila', 31), ('Chitoria_ulupi', 31), ('Arhopala_paramuta', 28), ('Euthalia_niepelti', 28), ('Seseria_dohertyi', 20)]

# plot_bar(train_labels_nums)


"""测试集分布"""
label_list = []
accuarcy = []
losses = []
rs = open('./log/88.49%.log')
line = rs.readlines()
for i in range(430, len(line)-1):
    tmp = line[i]
    if not 'Current Pred' in tmp:
        label_list.append(tmp)

num_count = {}
for i in label_list:
    if i not in num_count:
        num_count[i] = 1
    else:
        num_count[i] += 1
num_count = sorted(num_count.items(), key=lambda i: i[1], reverse=True)
print(num_count)

# plot_bar(num_count)

"""绘制loss"""
for i in range(116, 625):
    tmp = line[i]
    if 'Accuracy' in tmp:
        tmp_acc = float(tmp.split(' ')[-1][:-2])
        accuarcy.append(tmp_acc)
        tmp = line[i-2]
        tmp_loss = float(tmp.split(' ')[-1])
        losses.append(tmp_loss)
print(accuarcy)
print(losses)

epoch = np.arange(0, len(accuarcy))
fig = plt.figure()
acc = fig.add_subplot(121)
loss = fig.add_subplot(122)
acc.plot(np.arange(0, len(accuarcy)), accuarcy)
loss.plot(np.arange(0, len(losses)), losses)
plt.show()