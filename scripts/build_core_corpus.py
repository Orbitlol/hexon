#!/usr/bin/env python3
"""
Build the hand-authored Hexon core knowledge corpus.

Writes `data/knowledge/hexon_core_science.jsonl` - verified, self-contained
reference passages used to ground Hexon in ChromaDB. Every passage is written
to be 200-300 words so it lands in the existing chunk schema without being
split, and every claim in it is a fact the author was willing to defend
without a citation. Nothing here is scraped: this is the "verifiable core"
half of the hexon-1.5 corpus.

Deliberately excluded: nothing is held back from this corpus. The evaluation
set is protected by the grounded/ungrounded split in `hexon.evalbench`, not by
crippling the knowledge base - a model that cannot look up the capital of
Australia is not a better model, it is just a less useful one.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "knowledge" / "hexon_core_science.jsonl"

SCIENCE = [
    {
        "id": "core-sci-001",
        "title": "Motion, Force and Newton's Laws",
        "category": "science",
        "text": """Motion is described by position, velocity and acceleration. Velocity is the rate of change of position and includes a direction, whereas speed does not. Acceleration is the rate of change of velocity, so an object moving at constant speed in a circle is still accelerating because its direction keeps changing. Sir Isaac Newton published three laws that describe how forces produce motion. The first law states that an object stays at rest, or keeps moving in a straight line at constant speed, unless a net external force acts on it; this property is called inertia. The second law quantifies it: net force equals mass times acceleration, written F equals m a, so a heavy object needs a larger force to accelerate at the same rate as a light one. The third law states that whenever one object exerts a force on a second, the second exerts an equal and opposite force back. Friction and air resistance are forces that oppose motion and are why a sliding box eventually stops. Gravity is an attractive force between masses; near the Earth's surface it gives falling objects an acceleration of about 9.8 metres per second squared, which is why heavy and light objects fall at the same rate when air resistance is negligible.""",
    },
    {
        "id": "core-sci-002",
        "title": "Energy, Work and Power",
        "category": "science",
        "text": """Energy is the ability to do work, and it is measured in joules. Work is done when a force moves an object: work equals force multiplied by the distance moved in the direction of the force. Energy is conserved, meaning it cannot be created or destroyed, only converted from one store to another. Kinetic energy is the energy of movement and depends on mass and the square of speed, so doubling the speed of a car quadruples its kinetic energy. Gravitational potential energy is stored when an object is raised above the ground and depends on its mass, the height and the gravitational field strength. Chemical energy is stored in fuels, food and batteries. Thermal energy is the random motion of particles and flows from hotter regions to colder ones. Electrical energy is carried by moving charges. Power is the rate at which energy is transferred or work is done, measured in watts, where one watt is one joule per second. A 100 watt bulb converts 100 joules of electrical energy into light and heat every second. No real conversion is perfectly efficient: some energy is always dissipated as heat, which is why machines warm up during use.""",
    },
    {
        "id": "core-sci-003",
        "title": "Light, Colour and the Electromagnetic Spectrum",
        "category": "science",
        "text": """Light is electromagnetic radiation that travels at about 300 million metres per second in a vacuum. The electromagnetic spectrum runs from long wavelength radio waves through microwaves, infrared, visible light, ultraviolet, X-rays and finally short wavelength gamma rays. Visible light occupies a narrow band of wavelengths between roughly 400 and 700 nanometres, and we perceive different wavelengths within that band as different colours, from violet at the short end to red at the long end. White light such as sunlight is a mixture of all visible wavelengths; a prism or a raindrop separates them because different wavelengths refract by slightly different amounts. When light strikes a surface it can be reflected, absorbed or transmitted. An object appears a particular colour because it reflects that part of the spectrum and absorbs the rest. The sky appears blue because of Rayleigh scattering: air molecules scatter short wavelength blue light far more strongly than the longer red wavelengths, so blue light reaching our eyes from across the sky has been scattered out of the direct sunlight beam. At sunset the light path through the atmosphere is much longer, most of the blue has been scattered away, and the remaining direct light looks red.""",
    },
    {
        "id": "core-sci-004",
        "title": "Atoms, Elements and the Periodic Table",
        "category": "science",
        "text": """Every substance is made of atoms. An atom has a small dense nucleus containing positively charged protons and neutral neutrons, surrounded by negatively charged electrons. The number of protons is the atomic number and defines which element the atom is; hydrogen has one proton, carbon has six, iron has 26 and oxygen has eight. The total number of protons plus neutrons is the mass number. Atoms of the same element with different numbers of neutrons are called isotopes. The periodic table arranges elements in order of increasing atomic number into rows called periods and columns called groups, and elements in the same group share similar chemical behaviour. Each element has a one or two letter chemical symbol, usually derived from its English or Latin name: H for hydrogen, O for oxygen, C for carbon, and Fe for iron, from the Latin ferrum. Other examples include Na for sodium, K for potassium, from kalium, and Au for gold, from aurum. Elements are broadly divided into metals on the left and centre of the table and non metals on the upper right, with a diagonal band of metalloids between them. Noble gases in the final group are very unreactive because their outer electron shells are full.""",
    },
    {
        "id": "core-sci-005",
        "title": "Chemical Bonding, Reactions and Equations",
        "category": "science",
        "text": """Atoms combine by forming chemical bonds, which involve the outer shell electrons. In ionic bonding one atom transfers electrons to another, producing oppositely charged ions that attract each other; sodium chloride, ordinary table salt with the formula NaCl, is the classic example, formed from sodium ions and chloride ions in a cubic crystal lattice. In covalent bonding atoms share pairs of electrons, as in water, H2O, or carbon dioxide, CO2. Metallic bonding consists of positive metal ions held together by a sea of delocalised electrons, which is why metals conduct electricity and can be bent without shattering. A chemical reaction rearranges atoms into new substances and is written as an equation with reactants on the left and products on the right. Mass is conserved, so equations must be balanced, with the same number of each type of atom on both sides. Reactions release energy, called exothermic, as in combustion, or absorb energy, called endothermic. Reaction rates increase with temperature, concentration, surface area and the presence of a catalyst, which speeds up a reaction without being used up itself.""",
    },
    {
        "id": "core-sci-006",
        "title": "Water, Solutions, Acids and Bases",
        "category": "science",
        "text": """Water is a polar molecule, meaning the oxygen end carries a slight negative charge and the hydrogen ends a slight positive charge. This polarity makes water an excellent solvent, which is why it is called the universal solvent and why transport in living organisms depends on it. Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius at sea level; both temperatures fall at higher altitude because atmospheric pressure is lower. A solution forms when a solute dissolves in a solvent, and its concentration measures how much solute is present in a given volume. The pH scale runs from 0 to 14 and measures how acidic or alkaline a solution is. A pH below 7 is acidic, 7 is neutral, and above 7 is alkaline. Acids release hydrogen ions in water and include hydrochloric acid, sulfuric acid and citric acid; bases accept hydrogen ions and include sodium hydroxide and ammonia. When an acid and a base react they neutralise each other, producing a salt and water. Indicators such as litmus and universal indicator change colour to reveal the pH of a solution.""",
    },
    {
        "id": "core-sci-007",
        "title": "Cells, Organelles and Living Organisation",
        "category": "science",
        "text": """The cell is the basic unit of life. Plant and animal cells are eukaryotic, meaning their genetic material is enclosed in a nucleus. The nucleus stores DNA and controls the cell's activities. The cytoplasm is the jelly like material where most chemical reactions happen, and it is enclosed by the cell membrane, which controls which substances enter and leave. Mitochondria carry out aerobic respiration, releasing energy from glucose, which is why they are called the powerhouse of the cell; cells that need lots of energy, such as muscle cells, contain many of them. Ribosomes build proteins. The endoplasmic reticulum folds and transports proteins, and the Golgi apparatus modifies and packages them for export. Lysosomes digest worn out cell parts. Plant cells additionally have a rigid cellulose cell wall for support, a large permanent vacuole that stores sap, and chloroplasts containing chlorophyll, which absorb light for photosynthesis. Living things are organised into cells, tissues, organs, organ systems and whole organisms. Bacteria are prokaryotic: they are much smaller and have no nucleus.""",
    },
    {
        "id": "core-sci-008",
        "title": "The Human Skeleton and Muscles",
        "category": "science",
        "text": """The human skeleton has several jobs: it supports the body, protects delicate organs, allows movement and makes blood cells. A newborn has around 270 bones, but many fuse together during growth, so the typical adult skeleton contains 206 bones. The skeleton divides into the axial skeleton, which is the skull, spine and rib cage, and the appendicular skeleton, which is the limbs along with the shoulder and hip girdles. The skull protects the brain, while the rib cage protects the heart and lungs and the spine protects the spinal cord. Bones are living tissue made of a hard matrix of calcium and phosphorus salts reinforced with collagen fibres, which makes them strong but slightly flexible. They are light because the shafts contain marrow cavities. Bone marrow produces red blood cells, white blood cells and platelets. Joints are where two or more bones meet; some are fixed, such as those in the skull, while synovial joints like the knee and shoulder are freely movable and contain lubricating synovial fluid and tough ligaments. Muscles move bones by contracting and pulling across joints, and because a muscle can only pull, muscles work in antagonistic pairs.""",
    },
    {
        "id": "core-sci-009",
        "title": "Blood, the Heart and Circulation",
        "category": "science",
        "text": """Blood is a tissue made of a liquid called plasma with cells suspended in it. Red blood cells, also called erythrocytes, are the most numerous and carry oxygen from the lungs to the body's tissues. They are biconcave discs with no nucleus, and they contain haemoglobin, an iron containing protein that binds oxygen in the lungs and releases it where oxygen is scarce, giving blood its red colour. White blood cells defend the body against infection: phagocytes engulf invading microbes and lymphocytes produce antibodies. Platelets are small cell fragments that help blood clot at a wound. Plasma transports dissolved nutrients, hormones, carbon dioxide and urea. The human heart is a muscular pump with four chambers: two atria on top and two ventricles below. The right side pumps deoxygenated blood to the lungs, and the left side pumps oxygenated blood around the body. Valves prevent backflow, and the heart's own muscle is supplied with blood by the coronary arteries. Blood leaves the heart through arteries, returns through veins, and exchanges materials with tissues through thin walled capillaries.""",
    },
    {
        "id": "core-sci-010",
        "title": "Photosynthesis and Plant Nutrition",
        "category": "science",
        "text": """Photosynthesis is the process by which green plants, algae and some bacteria make their own food using light energy. It takes place mainly in the leaves, inside chloroplasts, whose chlorophyll pigment absorbs the sunlight. The plant takes in carbon dioxide from the air through tiny pores called stomata and absorbs water from the soil through its roots. Using light energy, the plant converts the carbon dioxide and water into glucose, a sugar that stores chemical energy, and oxygen is released as a by product. The overall word equation is carbon dioxide plus water, in the presence of light and chlorophyll, gives glucose plus oxygen. The glucose is used immediately for respiration, converted into starch for storage, or built into cellulose for cell walls. Plants also need minerals from the soil: nitrates to make proteins and thus grow, phosphates for roots and flowers, potassium for healthy leaves and flowers, and magnesium to build chlorophyll. A shortage of any of these causes deficiency symptoms such as yellowing leaves or stunted growth. Because photosynthesis removes carbon dioxide from the air and releases oxygen, plants are the foundation of almost every food chain.""",
    },
    {
        "id": "core-sci-011",
        "title": "Earth's Atmosphere and Weather",
        "category": "science",
        "text": """The Earth's atmosphere is a mixture of gases held around the planet by gravity. By volume it is roughly 78 percent nitrogen, making nitrogen by far the most abundant gas, and about 21 percent oxygen, which is the gas that animals need for respiration. The remaining one percent is mostly argon, with a small and climatically important amount of carbon dioxide, around 0.04 percent, plus variable water vapour. The atmosphere is layered: weather happens in the lowest layer, the troposphere, while the stratosphere above it contains the ozone layer that absorbs harmful ultraviolet radiation. Weather is the state of the atmosphere at a particular place and time, described by temperature, humidity, wind, cloud cover and precipitation. Rain forms when water evaporates, cools, condenses into droplets around dust particles to make clouds, and the droplets grow heavy enough to fall. This is part of the water cycle, which also includes transpiration from plants and runoff back to the sea. Air pressure falls with altitude, and winds blow because air moves from regions of high pressure toward regions of low pressure.""",
    },
    {
        "id": "core-sci-012",
        "title": "Rocks, Minerals and Plate Tectonics",
        "category": "science",
        "text": """Minerals are naturally occurring crystalline solids with a definite chemical composition. Diamond is the hardest natural substance known, scoring 10 on the Mohs scale of hardness; it is a form of pure carbon in which every atom is bonded to four others in a rigid three dimensional lattice. Graphite is also pure carbon but has a layered structure, which makes it soft and slippery, illustrating that the same element can form very different materials. Quartz, feldspar and mica are common rock forming minerals. Rocks are aggregates of minerals and fall into three families. Igneous rocks such as granite and basalt form when magma or lava cools and solidifies. Sedimentary rocks such as sandstone, limestone and shale form from particles or shells that are deposited, buried and cemented together. Metamorphic rocks such as marble and slate form when existing rocks are changed by heat and pressure. The Earth's outer shell, the lithosphere, is broken into tectonic plates that drift slowly on the mantle below. Where plates pull apart, magma rises to form new crust; where they collide, mountain ranges rise; and where one plate slides beneath another, earthquakes and volcanic eruptions result.""",
    },
    {
        "id": "core-sci-013",
        "title": "Human Nutrition and Digestion",
        "category": "science",
        "text": """A balanced diet supplies carbohydrates for energy, proteins for growth and repair, fats for energy storage and insulation, vitamins and mineral ions in small amounts for health, fibre to keep the digestive system working, and water for every cellular process. Carbohydrates include sugars and starches, found in bread, rice, pasta and potatoes. Proteins are built from amino acids and are abundant in meat, fish, eggs, beans and lentils. Fats come from oils, butter, nuts and oily fish. Vitamin C prevents scurvy and comes from citrus fruit and fresh vegetables, vitamin D is needed to absorb calcium and is made in the skin in sunlight, and iron is needed to make haemoglobin, with red meat and leafy greens being good sources. Digestion breaks large insoluble molecules into small soluble ones that can be absorbed into the blood. It begins in the mouth, where amylase starts digesting starch. In the stomach, hydrochloric acid kills bacteria and protease begins digesting protein. In the small intestine, bile from the liver neutralises acid and emulsifies fats, while enzymes from the pancreas and intestinal wall complete digestion. The villi lining the small intestine give it a huge surface area for absorption, and the large intestine then absorbs water back into the body.""",
    },
]

GEOGRAPHY = [
    {
        "id": "core-geo-001",
        "title": "Countries, Capitals and Territories",
        "category": "geography",
        "text": """A capital city is the seat of a country's government, and it is often, but by no means always, the largest city. Paris is the capital of France, Berlin is the capital of Germany, Madrid is the capital of Spain and Rome is the capital of Italy. Tokyo is the capital of Japan. Beijing is the capital of China and New Delhi is the capital of India. Some capitals surprise people precisely because they are not the biggest city. Canberra is the capital of Australia, chosen as a compromise between the rival cities of Sydney and Melbourne and purpose built for the role. Ottawa is the capital of Canada, not Toronto or Montreal, and it sits on the Ontario side of the border with Quebec. Washington, D.C. is the capital of the United States, while New York is its largest city. Brasília is the inland purpose built capital of Brazil, and Ankara is the capital of Turkey rather than Istanbul. Bern is the Swiss capital, not Zurich or Geneva. Some territories have more than one capital: South Africa uniquely has three, with Pretoria, Cape Town and Bloemfontein hosting different branches of government.""",
    },
    {
        "id": "core-geo-002",
        "title": "Oceans, Seas and Coastlines",
        "category": "geography",
        "text": """About 71 percent of the Earth's surface is covered by salt water, and that water is divided among five named oceans. The Pacific Ocean is by far the largest and deepest, covering more area than all of the land on Earth put together and containing the Mariana Trench, the deepest known point in the seabed at roughly 11 kilometres down. The Atlantic Ocean is the second largest and separates the Americas from Europe and Africa; the widest point of the Atlantic is much narrower than the Pacific, which is why transatlantic crossing came first. The Indian Ocean is the third largest, bounded by Africa, Asia, Australia and the Southern Ocean. The Southern Ocean, also called the Antarctic Ocean, encircles Antarctica and is defined by the strong Antarctic Circumpolar Current. The Arctic Ocean is the smallest and shallowest and is largely covered by sea ice. Seas are smaller bodies of salt water partly enclosed by land, such as the Mediterranean Sea, the Caribbean Sea, the Red Sea and the South China Sea. Ocean currents transport heat around the planet and strongly influence regional climates, which is why northwest Europe is far milder than other places at the same latitude.""",
    },
    {
        "id": "core-geo-003",
        "title": "Rivers, Lakes and Fresh Water",
        "category": "geography",
        "text": """Rivers are natural flowing watercourses that usually drain into an ocean, a sea, a lake or another river. The Amazon River in South America carries more water than any other river on Earth and drains the largest river basin in the world, running from the Peruvian Andes across Brazil to the Atlantic. Its length is close to that of the Nile in Africa, and which of the two is the longest river in the world is genuinely disputed among geographers, so it is safer to say the Amazon is the largest by volume while the Nile is traditionally cited as the longest. Other great rivers include the Yangtze in China, the Mississippi and Missouri system in North America, the Yenisei in Russia, the Congo in central Africa and the Ganges in South Asia, which is sacred in Hinduism. A river's drainage basin is the whole area of land whose water flows into it. Lakes are bodies of standing water surrounded by land. The Caspian Sea is the largest lake in the world by area but is salty, so Lake Superior counts as the largest freshwater lake by surface area, while Lake Baikal in Siberia holds the largest volume of fresh water and is also the deepest lake.""",
    },
    {
        "id": "core-geo-004",
        "title": "Mountains, Ranges and High Places",
        "category": "geography",
        "text": """Mountains are landforms that rise prominently above their surroundings, usually with steep slopes and a summit. Most of the world's great ranges formed where tectonic plates collided and pushed the crust upward. The Himalayas are the highest mountain range on Earth and run across Nepal, India, Bhutan, China and Pakistan. They contain Mount Everest, which at 8849 metres is the highest peak above sea level in the world; it sits on the border between Nepal and China's Tibet region and was first summited by Sir Edmund Hillary and Tenzing Norgay in 1953. K2, on the Pakistan and China border, is the second highest and is widely considered the harder and more dangerous climb. Outside Asia, the Andes form the longest continental mountain range, running down the western edge of South America, with Aconcagua its highest peak. The Rocky Mountains run down western North America, the Alps across central Europe, and the Great Dividing Range along eastern Australia. Africa's highest mountain is Mount Kilimanjaro in Tanzania, a free standing volcano. Mountains affect climate by forcing air upward, cooling it and producing rain on the windward side, leaving the far side dry.""",
    },
    {
        "id": "core-geo-005",
        "title": "Deserts, Climates and Biomes",
        "category": "geography",
        "text": """A desert is defined by aridity rather than by heat: it is a region that receives very little precipitation, typically under 250 millimetres a year. By that measure Antarctica is technically the largest desert on Earth, but the Sahara in North Africa is the largest hot desert, covering an area comparable to the United States across much of northern Africa. Other major hot deserts include the Arabian Desert, the Gobi in Mongolia and China, the Kalahari and Namib in southern Africa, the Thar in India and Pakistan, the Mojave and Sonoran in North America, and the Atacama in Chile, which is one of the driest places on Earth. Deserts form where dry air descends, around 30 degrees north and south of the equator, where rain bearing winds rarely reach, or in the rain shadow of a mountain range. Plants and animals there adapt with water storage, deep roots, reflective surfaces and nocturnal habits. Major land biomes besides desert include tropical rainforest, savanna, temperate grassland, Mediterranean scrub, temperate forest, boreal forest or taiga, and tundra.""",
    },
    {
        "id": "core-geo-006",
        "title": "Continents, Countries and Extremes",
        "category": "geography",
        "text": """The Earth's land is divided among seven continents: Asia, Africa, North America, South America, Antarctica, Europe and Oceania. Asia is the largest by both area and population and contains the highest point on Earth, Mount Everest. Africa is the second largest and has the longest north south extent, spanning the equator. Antarctica is the coldest, driest and windiest continent and holds most of the planet's fresh water as ice. Oceania is the smallest continent and is dominated by Australia. There are around 195 countries in the world. The largest by land area are Russia, Canada, China, the United States and Brazil. The most populous are India and China, each with well over a billion people, followed by the United States, Indonesia and Pakistan. The smallest sovereign state in the world by both area and population is Vatican City, an enclave inside Rome covering less than half a square kilometre and home to a few hundred people; it is the headquarters of the Roman Catholic Church. Other very small states include Monaco, Nauru, Tuvalu and San Marino.""",
    },
]

HISTORY = [
    {
        "id": "core-his-001",
        "title": "The Ancient World",
        "category": "history",
        "text": """The first civilisations appeared where fertile river valleys supported surplus farming. Sumer arose inMesopotamia between the Tigris and Euphrates rivers around 4000 BC, developing cuneiform writing, wheeled vehicles and city states. Ancient Egypt grew along the Nile from around 3100 BC, ruled by pharaohs and famous for pyramid building, mummification and hieroglyphic writing. The Indus Valley civilisation flourished in what is now Pakistan and northwest India with well planned grid cities. In China, the Shang dynasty developed bronze casting and early Chinese script. Classical Greece, from around 800 BC, gave the world democracy in Athens, the Olympic Games, the philosophy of Socrates, Plato and Aristotle, and the historical and scientific foundations of Western thought. Alexander the Great spread Greek culture across Persia and Egypt in the fourth century BC. Rome began as a small city state, became a republic, and under the emperors controlled the entire Mediterranean world. Roman engineering produced roads, aqueducts and concrete. The Roman Empire split in 395 AD, and the western half fell in 476 AD, beginning the Middle Ages.""",
    },
    {
        "id": "core-his-002",
        "title": "The Middle Ages and the Renaissance",
        "category": "history",
        "text": """The Middle Ages, or medieval period, ran roughly from the fall of Rome in the fifth century to the fifteenth century. Europe was largely rural and organised around manors and the feudal system, in which land was held in exchange for military service. The Christian Church was the dominant institution, and monasteries preserved literacy. The Black Death, a plague pandemic in the mid fourteenth century, killed a very large share of Europe's population and transformed its economy. The Islamic world preserved and extended classical science, medicine and mathematics through this period. The Renaissance began in Italy in the fourteenth and fifteenth centuries as a revival of classical learning and an explosion of art and science. Leonardo da Vinci painted the Mona Lisa and filled notebooks with engineering designs, while Michelangelo sculpted David and painted the Sistine Chapel ceiling. Johannes Gutenberg's movable type printing press, developed around 1450, made books cheap for the first time. Then came the Age of Exploration: Columbus reached the Americas in 1492, and Magellan's expedition circumnavigated the globe, completing the voyage in 1522 after his death.""",
    },
    {
        "id": "core-his-003",
        "title": "The World Wars and the Twentieth Century",
        "category": "history",
        "text": """The First World War began in 1914 after the assassination of Archduke Franz Ferdinand and ended in 1918, introducing industrialised trench warfare, tanks, poison gas and aircraft to conflict. The Treaty of Versailles imposed harsh terms on Germany. The Second World War began in 1939 when Germany invaded Poland and drew in Britain and France; it widened after Germany attacked the Soviet Union in 1941 and after Japan attacked Pearl Harbor, bringing the United States in. The war in Europe ended in May 1945 with Germany's surrender, and the war as a whole ended in September 1945 after Japan's surrender following the atomic bombings of Hiroshima and Nagasaki. It was the deadliest conflict in history. The Holocaust was the systematic murder of six million Jews by Nazi Germany. After 1945 the United States and the Soviet Union emerged as superpowers and entered the Cold War, competing through proxy wars, nuclear arms and the space race. The Berlin Wall, built in 1961 to divide East and West Berlin, was opened in November 1989, and the Soviet Union dissolved in 1991, ending the Cold War.""",
    },
    {
        "id": "core-his-004",
        "title": "The Space Age",
        "category": "history",
        "text": """The space age opened on 4 October 1957, when the Soviet Union launched Sputnik 1, the first artificial satellite. In April 1961 the Soviet cosmonaut Yuri Gagarin became the first human to travel into space and orbit the Earth. President John F. Kennedy then committed the United States to landing a person on the Moon before the end of the decade. That goal was met by the Apollo programme. On 20 July 1969, Apollo 11 landed in the Sea of Tranquility, and Neil Armstrong became the first person to walk on the Moon, followed by Buzz Aldrin, while Michael Collins orbited above. Armstrong's first words on stepping onto the surface described it as one small step for a man, one giant leap for mankind. Five further crewed landings followed through 1972. Later milestones included Skylab, the Apollo Soyuz handshake in 1975, the Space Shuttle programme from 1981, the Soviet Mir station, and the International Space Station, continuously crewed since November 2000. Uncrewed probes have now visited every planet, and the Voyager spacecraft have left the solar system entirely.""",
    },
    {
        "id": "core-his-005",
        "title": "Famous Ships, Disasters and Exploration Records",
        "category": "history",
        "text": """RMS Titanic was a British passenger liner operated by the White Star Line and, when she entered service, the largest ship afloat. On her maiden voyage from Southampton to New York she struck an iceberg in the North Atlantic late on 14 April 1912 and sank in the early hours of 15 April 1912. More than 1500 of the roughly 2200 people aboard died, in part because the ship carried lifeboats for only about half of those on board. The disaster led directly to major reforms, including the International Convention for the Safety of Life at Sea, better lifeboat provision and the establishment of a permanent ice patrol. The wreck was located in 1985 by Robert Ballard, lying about 3800 metres down. Other notable maritime events include the sinking of the Lusitania in 1915, which influenced American entry into the First World War, and the loss of the Bismarck in 1941. Earlier exploration records include Christopher Columbus reaching the Caribbean in 1492, Vasco da Gama reaching India by sea in 1498, and Ferdinand Magellan's expedition completing the first circumnavigation of the Earth in 1522.""",
    },
    {
        "id": "core-his-006",
        "title": "The Industrial Revolution and Modern Technology",
        "category": "history",
        "text": """The Industrial Revolution began in Britain in the late eighteenth century and transformed how goods were made. Mechanised textile spinning, powered first by water and then by the steam engine improved by James Watt, moved production from homes to factories. Iron production was revolutionised by coke smelting, and railways and steamships shrank travel times. The revolution spread to Europe and the United States through the nineteenth century, driving urbanisation and raising output while also creating harsh working conditions that eventually produced labour laws and unions. The Second Industrial Revolution, from about 1870, added steel, electricity, the internal combustion engine, chemicals and the telegraph and telephone. The twentieth century brought aviation, with the Wright brothers' first powered flight in 1903; antibiotics, with penicillin discovered by Alexander Fleming in 1928 and developed into a drug in the 1940s; nuclear power; the transistor, invented at Bell Labs in 1947; and integrated circuits. The Internet grew from ARPANET in 1969, and the personal computer and mobile phone put computing in billions of pockets.""",
    },
]

GENERAL = [
    {
        "id": "core-gen-001",
        "title": "Money, Currency and Trade",
        "category": "general",
        "text": """Every country issues its own currency, and exchange rates set how much one is worth in another. The yen is the currency of Japan, issued by the Bank of Japan and denoted by the symbol with the code JPY. Other major currencies include the United States dollar, the euro used by many European Union members, the British pound sterling, the Swiss franc and the Chinese yuan, also called the renminbi. In South Asia, India uses the rupee, Pakistan the Pakistani rupee, Bangladesh the taka and Sri Lanka the Sri Lankan rupee. The South Korean currency is the won. Historically the pound sterling was the dominant world currency during the nineteenth century, and the United States dollar has held that role since the mid twentieth century, serving as the main reserve currency held by central banks and the usual unit for pricing oil and many commodities. Inflation is the rate at which the general price level rises, eroding purchasing power, and central banks manage it mainly by setting interest rates. A weaker currency makes a country's exports cheaper abroad but imports more expensive at home.""",
    },
    {
        "id": "core-gen-002",
        "title": "Famous Books and Their Authors",
        "category": "general",
        "text": """The Hobbit is a children's fantasy novel written by J. R. R. Tolkien and published in 1937; it follows the reluctant hobbit Bilbo Baggins on a quest to reclaim a lost dwarf kingdom and the treasure guarded by the dragon Smaug, and it introduced the One Ring that drives the plot of the later Lord of the Rings. Tolkien was an Oxford professor of Anglo Saxon who built entire languages for his world. Other landmarks of English literature include Mary Shelley's Frankenstein from 1818, a founding work of science fiction; Jane Austen's Pride and Prejudice from 1813; Charles Dickens's A Tale of Two Cities and Great Expectations; the Sherlock Holmes stories of Arthur Conan Doyle; the Alice books of Lewis Carroll; Herman Melville's Moby Dick; and Virginia Woolf's Mrs Dalloway. George Orwell wrote Nineteen Eighty Four and Animal Farm, both political dystopias. Agatha Christie became the best selling novelist of all time through her detective fiction. In the modern era, J. K. Rowling's Harry Potter series and George R. R. Martin's A Song of Ice and Fire have reached enormous audiences.""",
    },
    {
        "id": "core-gen-003",
        "title": "Mathematics: Numbers, Percentages and Patterns",
        "category": "math",
        "text": """A percentage expresses a number as a fraction of 100, so 15 percent means 15 out of every 100. To find a percentage of a quantity, multiply the quantity by the percentage and divide by 100: 15 percent of 200 is 15 times 200 divided by 100, which is 30. Percentages above 100 describe growth beyond the original. A number sequence follows a rule. In the sequence 2, 4, 8, 16 each term is double the previous one, so the next term is 32; this is a geometric progression with a common ratio of 2. In an arithmetic progression the terms differ by a fixed amount, as in 3, 7, 11, 15 where the common difference is 4. A prime number is a whole number greater than 1 whose only positive divisors are 1 and itself, so 2, 3, 5, 7, 11 and 13 are prime while 1 is not and 6 is not because it divides by 2 and 3. Every whole number greater than 1 is either prime or can be written as a unique product of primes. The square root of 144 is 12 because 12 times 12 is 144, and 7 times 8 is 56.""",
    },
    {
        "id": "core-gen-004",
        "title": "Geometry and Measurement",
        "category": "math",
        "text": """Geometry studies points, lines, angles, surfaces and solids. An angle is measured in degrees, and a full turn is 360 degrees. A right angle is 90 degrees and a straight line is 180 degrees. The three interior angles of any triangle always add up to 180 degrees, which is why a triangle can never contain more than one right angle. Triangles are classified as equilateral with three equal sides and three 60 degree angles, isosceles with two equal sides, or scalene with no equal sides. The area of a rectangle is length times width, and the area of a triangle is half of its base times its perpendicular height. For a circle, the circumference is two times pi times the radius and the area is pi times the radius squared, where pi is approximately 3.14159. Pythagoras's theorem applies to right angled triangles: the square of the longest side, the hypotenuse, equals the sum of the squares of the other two sides. Solids are measured by volume, in cubic units, and a cube with side s has a volume of s cubed and a surface area of six s squared.""",
    },
]


# Passages are authored at roughly 170-210 words each. The ChromaDB chunker
# only splits documents ABOVE 300 words, so passages are merged into topic
# articles of 350-750 words; the chunker then emits clean 200-300 word pieces.
# Each group below is one article, and every article is verified against the
# real chunker by `main()`.

TOPIC_GROUPS = [
    ("Physics: Motion, Energy and Light", "science",
     ["core-sci-001", "core-sci-002", "core-sci-003"]),
    ("Chemistry: Atoms, Bonding and Reactions", "science",
     ["core-sci-004", "core-sci-005", "core-sci-006"]),
    ("Cells, Plant Nutrition and Digestion", "science",
     ["core-sci-007", "core-sci-010", "core-sci-013"]),
    ("The Human Body: Skeleton and Circulation", "science",
     ["core-sci-008", "core-sci-009"]),
    ("Earth Science: Atmosphere, Weather and Rocks", "science",
     ["core-sci-011", "core-sci-012"]),
    ("World Geography: Countries, Capitals and Continents", "geography",
     ["core-geo-001", "core-geo-006"]),
    ("World Geography: Oceans, Rivers and Lakes", "geography",
     ["core-geo-002", "core-geo-003"]),
    ("World Geography: Mountains, Deserts and Biomes", "geography",
     ["core-geo-004", "core-geo-005"]),
    ("History: The Ancient World to the Renaissance", "history",
     ["core-his-001", "core-his-002", "core-his-006"]),
    ("History: The World Wars, the Space Age and Disasters", "history",
     ["core-his-003", "core-his-004", "core-his-005"]),
    ("Money, Currency and Famous Books", "general",
     ["core-gen-001", "core-gen-002"]),
    ("Mathematics: Numbers, Patterns and Geometry", "math",
     ["core-gen-003", "core-gen-004"]),
]


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT))
    from hexon.rag import chunk_text

    OUT.parent.mkdir(parents=True, exist_ok=True)

    by_id = {d["id"]: d for d in (SCIENCE + GEOGRAPHY + HISTORY + GENERAL)}

    articles = []
    for title, category, ids in TOPIC_GROUPS:
        parts = [by_id[i] for i in ids if i in by_id]
        body = "\n\n".join(p["text"].strip() for p in parts)
        heading = f"{title}. " + ". ".join(p["title"] for p in parts) + "."
        articles.append(
            {
                "id": "core-art-" + ids[0].split("-", 1)[1].replace("-", "")[:6] + "-" + ids[0][-3:],
                "title": title,
                "category": category,
                "text": heading + "\n\n" + body,
                "source": "hexon_core_authored",
                "member_passages": ids,
            }
        )

    # de-duplicate ids
    seen = {}
    for a in articles:
        base = a["id"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n:
            a["id"] = f"{base}-{n}"

    with OUT.open("w", encoding="utf-8") as f:
        for a in articles:
            a["word_count"] = len(a["text"].split())
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Verify every article chunks into pieces that all sit inside 200-300 words.
    all_chunk_words: List[int] = []
    bad = []
    for a in articles:
        pieces = chunk_text(a["text"], min_words=200, max_words=300, overlap_words=35)
        for p in pieces:
            w = len(p.split())
            all_chunk_words.append(w)
            if not (200 <= w <= 300):
                bad.append((a["id"], w))

    counts: Dict[str, int] = {}
    for a in articles:
        counts[a["category"]] = counts.get(a["category"], 0) + 1

    print(f"wrote {len(articles)} articles -> {OUT.relative_to(ROOT)}")
    print(f"  categories  : {counts}")
    aw = [a["word_count"] for a in articles]
    print(f"  article len : min={min(aw)} max={max(aw)} avg={sum(aw)/len(aw):.1f}")
    print(f"  total chunks: {len(all_chunk_words)} "
          f"(min={min(all_chunk_words)} max={max(all_chunk_words)} "
          f"avg={sum(all_chunk_words)/len(all_chunk_words):.1f})")
    print(f"  chunks outside 200-300 words: {bad if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
