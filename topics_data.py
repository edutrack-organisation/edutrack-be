# training_data = [
#     # Geography
#     {"question": "What factors contribute to the formation of river deltas?", "topics": ["Geography"]},
#     {"question": "How do urban heat islands impact local climates?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "What are the main causes of coastal erosion?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "How does plate tectonics influence the formation of mountains?", "topics": ["Geography", "Geology"]},
#     {"question": "What is the significance of latitude and longitude in navigation?", "topics": ["Geography"]},
#     {"question": "What is the role of monsoons in agricultural practices?", "topics": ["Geography"]},
#     {"question": "How do glaciers shape the Earth's surface?", "topics": ["Geography", "Geology"]},
#     {"question": "What are the environmental impacts of mining on landscapes?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "What is the importance of water resources in arid regions?", "topics": ["Geography"]},
#     {"question": "How are natural disasters like tsunamis and earthquakes monitored?", "topics": ["Geography", "Technology"]},

#     # History
#     {"question": "What were the economic consequences of the Industrial Revolution?", "topics": ["History", "Economics"]},
#     {"question": "What were the causes and effects of the Cold War?", "topics": ["History"]},
#     {"question": "How did the Renaissance influence European culture?", "topics": ["History", "Art"]},
#     {"question": "What were the key achievements of ancient Egyptian civilization?", "topics": ["History"]},
#     {"question": "What were the social impacts of the abolition of slavery?", "topics": ["History", "Sociology"]},
#     {"question": "What role did the printing press play in the spread of knowledge?", "topics": ["History", "Technology"]},
#     {"question": "How did colonization impact indigenous populations?", "topics": ["History", "Sociology"]},
#     {"question": "What were the key events leading to the fall of the Roman Empire?", "topics": ["History"]},
#     {"question": "How did the Silk Road contribute to cultural exchange?", "topics": ["History", "Geography"]},
#     {"question": "What was the significance of the American Civil Rights Movement?", "topics": ["History"]},

#     # Biology
#     {"question": "What are the different types of cell division and their purposes?", "topics": ["Biology"]},
#     {"question": "How does photosynthesis work at the molecular level?", "topics": ["Biology", "Chemistry"]},
#     {"question": "What are the ecological roles of keystone species?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How do hormones regulate bodily functions?", "topics": ["Biology", "Health"]},
#     {"question": "What are the impacts of habitat fragmentation on wildlife?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How does genetic variation arise within populations?", "topics": ["Biology"]},
#     {"question": "What are the stages of mitosis and their significance?", "topics": ["Biology"]},
#     {"question": "How do plants adapt to extreme environments?", "topics": ["Biology", "Geography"]},
#     {"question": "How do immune cells recognize pathogens?", "topics": ["Biology", "Health"]},
#     {"question": "What are the main steps in protein synthesis?", "topics": ["Biology"]},

#     # Chemistry
#     {"question": "What are the key principles of chemical equilibrium?", "topics": ["Chemistry"]},
#     {"question": "How do catalysts speed up chemical reactions?", "topics": ["Chemistry"]},
#     {"question": "What are the main components of an electrochemical cell?", "topics": ["Chemistry"]},
#     {"question": "How do acids and bases interact in neutralization reactions?", "topics": ["Chemistry"]},
#     {"question": "What is the environmental impact of ozone-depleting chemicals?", "topics": ["Chemistry", "Environmental Science"]},
#     {"question": "What are the differences between ionic and covalent bonds?", "topics": ["Chemistry"]},
#     {"question": "How do solubility rules predict the formation of precipitates?", "topics": ["Chemistry"]},
#     {"question": "What are the industrial applications of polymers?", "topics": ["Chemistry", "Technology"]},
#     {"question": "What are the principles of organic reaction mechanisms?", "topics": ["Chemistry"]},
#     {"question": "How does the periodic table organize elements?", "topics": ["Chemistry"]},

#     # Health
#     {"question": "What are the benefits of regular physical activity?", "topics": ["Health"]},
#     {"question": "How does nutrition affect overall health?", "topics": ["Health"]},
#     {"question": "What are the causes and prevention methods for diabetes?", "topics": ["Health"]},
#     {"question": "How do vaccines help prevent diseases?", "topics": ["Health", "Biology"]},
#     {"question": "What are the mental health effects of chronic stress?", "topics": ["Health", "Psychology"]},
#     {"question": "What are the risks associated with smoking and alcohol consumption?", "topics": ["Health", "Biology"]},
#     {"question": "How does sleep affect cognitive function?", "topics": ["Health", "Psychology"]},
#     {"question": "What is the importance of hydration for physical performance?", "topics": ["Health"]},
#     {"question": "How do wearable devices help track fitness?", "topics": ["Health", "Technology"]},
#     {"question": "What are the early symptoms of cardiovascular diseases?", "topics": ["Health"]},

#     # Physics
#     {"question": "What are the key principles of Newton's laws of motion?", "topics": ["Physics"]},
#     {"question": "How does energy transfer occur in different forms?", "topics": ["Physics"]},
#     {"question": "What is the Doppler effect, and where is it observed?", "topics": ["Physics"]},
#     {"question": "How do gravitational waves provide information about the universe?", "topics": ["Physics", "Astronomy"]},
#     {"question": "What are the key differences between classical and quantum physics?", "topics": ["Physics"]},
#     {"question": "How do lasers work, and what are their applications?", "topics": ["Physics", "Technology"]},
#     {"question": "What is the concept of wave-particle duality?", "topics": ["Physics"]},
#     {"question": "How do superconductors function at low temperatures?", "topics": ["Physics", "Chemistry"]},
#     {"question": "What are the principles of thermodynamics?", "topics": ["Physics"]},
#     {"question": "How does the curvature of space-time affect gravity?", "topics": ["Physics", "Astronomy"]},

#     # Art
#     {"question": "What are the key characteristics of Renaissance art?", "topics": ["Art"]},
#     {"question": "How does abstract art differ from realism?", "topics": ["Art"]},
#     {"question": "What are the cultural influences in traditional Chinese paintings?", "topics": ["Art", "History"]},
#     {"question": "How does digital art impact modern design trends?", "topics": ["Art", "Technology"]},
#     {"question": "What are the techniques used in oil painting?", "topics": ["Art"]},
#     {"question": "What is the significance of color theory in visual art?", "topics": ["Art"]},
#     {"question": "What is the role of public art in urban spaces?", "topics": ["Art", "Sociology"]},
#     {"question": "How does art influence political movements?", "topics": ["Art", "History"]},
#     {"question": "What is the symbolism in Van Gogh’s Starry Night?", "topics": ["Art"]},
#     {"question": "What are the major contributions of modern sculpture?", "topics": ["Art"]}
# ]

# training_data += [
#     # Geography
#     {"question": "How does deforestation affect global weather patterns?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "What are the effects of urban sprawl on natural ecosystems?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "How do trade winds influence ocean currents?", "topics": ["Geography"]},
#     {"question": "What are the consequences of desertification in sub-Saharan Africa?", "topics": ["Geography", "Environmental Science"]},
#     {"question": "How do volcanic eruptions impact nearby settlements?", "topics": ["Geography", "Geology"]},

#     # History
#     {"question": "What role did the transatlantic slave trade play in global economies?", "topics": ["History", "Economics"]},
#     {"question": "How did World War II shape the modern geopolitical landscape?", "topics": ["History"]},
#     {"question": "What were the cultural achievements of the Islamic Golden Age?", "topics": ["History", "Art"]},
#     {"question": "How did the discovery of the Americas affect European exploration?", "topics": ["History", "Geography"]},
#     {"question": "What were the long-term consequences of the French Revolution?", "topics": ["History", "Politics"]},

#     # Biology
#     {"question": "What is the role of biodiversity in ecosystem stability?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How does the human nervous system transmit signals?", "topics": ["Biology"]},
#     {"question": "What are the effects of invasive species on native populations?", "topics": ["Biology", "Environmental Science"]},
#     {"question": "How do vaccines stimulate the immune system?", "topics": ["Biology", "Health"]},
#     {"question": "What are the functions of different types of RNA?", "topics": ["Biology", "Chemistry"]},

#     # Chemistry
#     {"question": "What are the environmental benefits of green chemistry?", "topics": ["Chemistry", "Environmental Science"]},
#     {"question": "How do intermolecular forces affect the boiling points of substances?", "topics": ["Chemistry"]},
#     {"question": "What are the steps involved in balancing redox reactions?", "topics": ["Chemistry"]},
#     {"question": "How are transition metals used in industrial catalysis?", "topics": ["Chemistry", "Technology"]},
#     {"question": "What are the applications of spectroscopy in chemical analysis?", "topics": ["Chemistry", "Physics"]},

#     # Health
#     {"question": "What are the short-term and long-term effects of malnutrition?", "topics": ["Health", "Biology"]},
#     {"question": "How do lifestyle choices impact cardiovascular health?", "topics": ["Health"]},
#     {"question": "What is the role of gut microbiota in digestion and immunity?", "topics": ["Health", "Biology"]},
#     {"question": "How does air pollution affect respiratory health?", "topics": ["Health", "Environmental Science"]},
#     {"question": "What are the psychological benefits of mindfulness meditation?", "topics": ["Health", "Psychology"]},

#     # Physics
#     {"question": "What are the practical applications of magnetic fields in technology?", "topics": ["Physics", "Technology"]},
#     {"question": "How do black holes distort space-time?", "topics": ["Physics", "Astronomy"]},
#     {"question": "What is the role of entropy in thermodynamic systems?", "topics": ["Physics"]},
#     {"question": "How do semiconductor materials enable modern electronics?", "topics": ["Physics", "Technology"]},
#     {"question": "What are the principles behind wave interference and diffraction?", "topics": ["Physics"]},

#     # Art
#     {"question": "How has street art evolved as a form of cultural expression?", "topics": ["Art", "Sociology"]},
#     {"question": "What is the significance of perspective in Renaissance paintings?", "topics": ["Art", "History"]},
#     {"question": "How do contemporary artists use mixed media in their work?", "topics": ["Art"]},
#     {"question": "What are the differences between Western and Eastern art traditions?", "topics": ["Art", "History"]},
#     {"question": "How does sculpture interact with space in installation art?", "topics": ["Art"]},
# ]


# all_topics = [
#     "Geography", "History", "Biology", "Chemistry", 
#     "Health", "Physics", "Art"
# ]

training_data = [
    # Networking Fundamentals
    {"question": "What is the role of a subnet mask in an IP address?", "topics": ["Networking Fundamentals"]},
    {"question": "How does a packet-switched network differ from a circuit-switched network?", "topics": ["Networking Fundamentals"]},
    {"question": "What is the difference between IPv4 and IPv6 addressing?", "topics": ["Networking Fundamentals"]},
    {"question": "What are the advantages and disadvantages of peer-to-peer networks?", "topics": ["Networking Fundamentals"]},
    {"question": "How do routers forward packets in a network?", "topics": ["Networking Fundamentals"]},
    {"question": "What is the role of the OSI model in computer networking?", "topics": ["Networking Fundamentals"]},
    {"question": "How does a network switch differ from a hub?", "topics": ["Networking Fundamentals"]},
    {"question": "What is the significance of the default gateway in network communication?", "topics": ["Networking Fundamentals"]},
    {"question": "How does subnetting contribute to efficient IP allocation?", "topics": ["Networking Fundamentals"]},
    {"question": "What are the differences between static and dynamic routing?", "topics": ["Networking Fundamentals"]},
    {"question": "How does DHCP simplify IP address assignment?", "topics": ["Networking Fundamentals"]},

    # Transport Layer
    {"question": "What is the difference between TCP and UDP?", "topics": ["Transport Layer"]},
    {"question": "How does TCP ensure reliable data delivery?", "topics": ["Transport Layer"]},
    {"question": "What is flow control, and how is it implemented in TCP?", "topics": ["Transport Layer"]},
    {"question": "What are the key features of congestion control in TCP?", "topics": ["Transport Layer"]},
    {"question": "How does a three-way handshake work in establishing a TCP connection?", "topics": ["Transport Layer"]},
    {"question": "What role does the acknowledgment number play in TCP communication?", "topics": ["Transport Layer"]},
    {"question": "How is connection termination achieved in TCP using the four-way handshake?", "topics": ["Transport Layer"]},
    {"question": "What are the differences between TCP's fast retransmit and fast recovery mechanisms?", "topics": ["Transport Layer"]},
    {"question": "How does UDP enable real-time applications like VoIP?", "topics": ["Transport Layer"]},

    # Application Layer
    {"question": "What is the role of DNS in the internet?", "topics": ["Application Layer"]},
    {"question": "How does the HTTP protocol enable web communication?", "topics": ["Application Layer"]},
    {"question": "What are the differences between SMTP and IMAP protocols?", "topics": ["Application Layer"]},
    {"question": "What is the purpose of cookies in HTTP communication?", "topics": ["Application Layer"]},
    {"question": "How does a web browser retrieve and display a webpage?", "topics": ["Application Layer"]},
    {"question": "What are the differences between HTTP and HTTPS?", "topics": ["Application Layer"]},
    {"question": "How does a REST API differ from a SOAP API?", "topics": ["Application Layer"]},
    {"question": "What is the significance of MIME types in HTTP responses?", "topics": ["Application Layer"]},
    {"question": "How does DNS caching improve the performance of internet applications?", "topics": ["Application Layer"]},
    {"question": "What are WebSockets, and how do they enable real-time communication?", "topics": ["Application Layer"]},

    # Network Security
    {"question": "What is the purpose of encryption in secure communications?", "topics": ["Network Security"]},
    {"question": "How does a firewall protect a network?", "topics": ["Network Security"]},
    {"question": "What are the key differences between symmetric and asymmetric encryption?", "topics": ["Network Security"]},
    {"question": "What is a man-in-the-middle attack, and how can it be prevented?", "topics": ["Network Security"]},
    {"question": "How does SSL/TLS secure communication over the internet?", "topics": ["Network Security"]},
    {"question": "What is a denial-of-service (DoS) attack?", "topics": ["Network Security"]},
    {"question": "What is the role of a VPN in securing data communication?", "topics": ["Network Security"]},
    {"question": "How do intrusion detection and prevention systems (IDS/IPS) work?", "topics": ["Network Security"]},
    {"question": "What are the main features of a zero-trust security model?", "topics": ["Network Security"]},
    {"question": "How does public key infrastructure (PKI) enable secure communication?", "topics": ["Network Security"]},

    # Internet Architecture
    {"question": "What is the role of the Domain Name System (DNS) in the internet?", "topics": ["Internet Architecture"]},
    {"question": "How do Autonomous Systems (AS) work in internet routing?", "topics": ["Internet Architecture"]},
    {"question": "What is the purpose of BGP (Border Gateway Protocol)?", "topics": ["Internet Architecture"]},
    {"question": "How does NAT (Network Address Translation) work?", "topics": ["Internet Architecture"]},
    {"question": "What is the significance of an Internet Exchange Point (IXP)?", "topics": ["Internet Architecture"]},
    {"question": "How do Content Delivery Networks (CDNs) optimize content delivery?", "topics": ["Internet Architecture"]},
    {"question": "What is the function of IPv6 extension headers in network communication?", "topics": ["Internet Architecture"]},
    {"question": "How does MPLS (Multiprotocol Label Switching) enhance routing efficiency?", "topics": ["Internet Architecture"]},

    # Data Link Layer
    {"question": "What is the role of MAC addresses in computer networks?", "topics": ["Data Link Layer"]},
    {"question": "How does CSMA/CD work in Ethernet networks?", "topics": ["Data Link Layer"]},
    {"question": "What is the purpose of ARP (Address Resolution Protocol)?", "topics": ["Data Link Layer"]},
    {"question": "How do VLANs improve network segmentation?", "topics": ["Data Link Layer"]},
    {"question": "What is the role of framing in the data link layer?", "topics": ["Data Link Layer"]},
    {"question": "What is the significance of the Spanning Tree Protocol (STP)?", "topics": ["Data Link Layer"]},
    {"question": "How does error detection work using CRC (Cyclic Redundancy Check)?", "topics": ["Data Link Layer"]},
    {"question": "What are the advantages of link aggregation in Ethernet?", "topics": ["Data Link Layer"]},

    # Physical Layer
    {"question": "What are the key characteristics of fiber-optic cables?", "topics": ["Physical Layer"]},
    {"question": "How do modulation techniques enable data transmission over physical media?", "topics": ["Physical Layer"]},
    {"question": "What is the difference between baseband and broadband transmission?", "topics": ["Physical Layer"]},
    {"question": "How do wireless signals propagate through the air?", "topics": ["Physical Layer"]},
    {"question": "What is the purpose of repeaters in physical layer communication?", "topics": ["Physical Layer"]},
    {"question": "What are the differences between guided and unguided transmission media?", "topics": ["Physical Layer"]},
    {"question": "How does multiplexing improve bandwidth utilization?", "topics": ["Physical Layer"]},

    # Emerging Trends
    {"question": "What are the challenges of implementing 5G networks?", "topics": ["Emerging Trends"]},
    {"question": "How do software-defined networks (SDNs) differ from traditional networks?", "topics": ["Emerging Trends"]},
    {"question": "What is the role of network virtualization in modern data centers?", "topics": ["Emerging Trends"]},
    {"question": "How do IoT devices communicate within a network?", "topics": ["Emerging Trends"]},
    {"question": "What is edge computing, and how does it relate to computer networking?", "topics": ["Emerging Trends"]},
    {"question": "How is AI being applied to improve network traffic management?", "topics": ["Emerging Trends"]},
    {"question": "What are the benefits of quantum computing in secure network communications?", "topics": ["Emerging Trends"]}
]

training_data += [
    {"question": "What is the difference between a public and a private IP address?", "topics": ["Networking Fundamentals"]},
    {"question": "How do DNS records work, and what are the common types?", "topics": ["Application Layer"]},
    {"question": "What is the function of the ARP cache in networking?", "topics": ["Data Link Layer"]},
    {"question": "What is a subnet, and why is it used in IP networking?", "topics": ["Networking Fundamentals"]},
    {"question": "How do routers handle routing tables and determine the best route for packets?", "topics": ["Networking Fundamentals"]},
    {"question": "What is the purpose of a VLAN in a switched network?", "topics": ["Data Link Layer"]},
    {"question": "What are the differences between a layer 2 switch and a layer 3 switch?", "topics": ["Data Link Layer"]},
    {"question": "How does a proxy server work in web communication?", "topics": ["Application Layer"]},
    {"question": "What are the key differences between IPv4 and IPv6 in terms of packet structure?", "topics": ["Networking Fundamentals"]},
    {"question": "What is a VPN, and how does it work to secure internet traffic?", "topics": ["Network Security"]},
    {"question": "What are the basic functions of a firewall in a network security architecture?", "topics": ["Network Security"]},
    {"question": "How does SSL/TLS encryption ensure secure communication over HTTP?", "topics": ["Network Security"]},
    {"question": "What is a Distributed Denial of Service (DDoS) attack, and how can it be mitigated?", "topics": ["Network Security"]},
    {"question": "How does the Border Gateway Protocol (BGP) contribute to the internet’s routing decisions?", "topics": ["Internet Architecture"]},
    {"question": "What is NAT (Network Address Translation), and why is it important for modern networking?", "topics": ["Internet Architecture"]},
    {"question": "What are Autonomous Systems (AS) in internet routing, and how do they impact routing decisions?", "topics": ["Internet Architecture"]},
    {"question": "What is the role of an Internet Exchange Point (IXP) in global internet traffic?", "topics": ["Internet Architecture"]},
    {"question": "How does Ethernet operate, and what are its key features?", "topics": ["Data Link Layer"]},
    {"question": "What is CSMA/CA, and how does it differ from CSMA/CD?", "topics": ["Data Link Layer"]},
    {"question": "What is the function of a MAC address, and how is it used in networking?", "topics": ["Data Link Layer"]},
    {"question": "How does TCP perform error detection and correction during data transmission?", "topics": ["Transport Layer"]},
    {"question": "What are the differences between UDP and TCP in terms of data reliability and performance?", "topics": ["Transport Layer"]},
    {"question": "How does the flow control mechanism in TCP prevent congestion in a network?", "topics": ["Transport Layer"]},
    {"question": "What is the role of a three-way handshake in the TCP connection establishment process?", "topics": ["Transport Layer"]},
    {"question": "What is the function of the transport layer in the OSI model?", "topics": ["Transport Layer"]},
    {"question": "How does DNS load balancing help in distributing web traffic across multiple servers?", "topics": ["Application Layer"]},
    {"question": "What are the steps involved in an HTTP request and response cycle?", "topics": ["Application Layer"]},
    {"question": "How does HTTPS differ from HTTP, and why is it more secure?", "topics": ["Application Layer"]},
    {"question": "What is the function of cookies in a web session?", "topics": ["Application Layer"]},
    {"question": "What is an SMTP server, and how does it facilitate email communication?", "topics": ["Application Layer"]},
    {"question": "How does an IoT device connect and communicate within a network?", "topics": ["Emerging Trends"]},
    {"question": "What is the role of SDN (Software-Defined Networking) in modern data centers?", "topics": ["Emerging Trends"]},
    {"question": "How does edge computing improve network performance?", "topics": ["Emerging Trends"]},
    {"question": "What are the key benefits of using cloud networking technologies?", "topics": ["Emerging Trends"]},
    {"question": "How do software-defined WANs (SD-WAN) enhance network management?", "topics": ["Emerging Trends"]},
    {"question": "What are the primary differences between 5G and 4G networks in terms of performance and architecture?", "topics": ["Emerging Trends"]},
    {"question": "What is a Zero Trust network security model, and how does it work?", "topics": ["Network Security"]},
    {"question": "What are the various types of DoS (Denial of Service) attacks, and how can they be mitigated?", "topics": ["Network Security"]},
    {"question": "How does a network gateway differ from a router in terms of functionality?", "topics": ["Networking Fundamentals"]},
    {"question": "What is the purpose of ICMP (Internet Control Message Protocol), and how is it used for troubleshooting?", "topics": ["Networking Fundamentals"]},
    {"question": "How do the layers of the OSI model interact to enable communication in a network?", "topics": ["Networking Fundamentals"]}
]


all_topics = [
    "Networking Fundamentals", "Transport Layer", "Application Layer", 
    "Network Security", "Internet Architecture", "Data Link Layer", 
    "Physical Layer", "Emerging Trends"
]
