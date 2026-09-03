
ICCAD 2025 Problem A - Trojan Type Summary
------------------------------------------

Trojan0 - Key Leakage via XOR with LFSR
- Description: Leaks the secret 128-bit key by XORing segments of it with a 20-bit LFSR output.
- Behavior: Information leakage
- Trigger: Clock driven
- Output: load[63:0] ← key[i/8] ^ counter[i/8]

Trojan1 - Event Counter Trigger
- Description: Uses a 4-bit internal counter that increments when `r1 == 1`. When MSB is high, triggers output.
- Behavior: Trigger event
- Trigger: r1 == 1 repeatedly
- Output: trigger = counter[3]

Trojan2 - Pattern-Based Force Reset
- Description: Detects a specific 2-pattern input change (0xAA → 0x55) to activate reset signal.
- Behavior: Trigger event → force reset
- Trigger: prev_data == 0xAA and data_in == 0x55
- Output: force_reset

Trojan3 - Conditional Logic Modification
- Description: On 256th clock cycle (counter==255), modifies data output by adding 2.
- Behavior: Selective logic modification
- Trigger: Counter overflow
- Output: data_out = data_in + 2

Trojan4 - Key Leakage via XOR with LFSR Pattern
- Description: Uses a 16-bit LFSR and replicates it to mask the key, leaking data via XOR.
- Behavior: Information leakage
- Trigger: Clock driven
- Output: leak = key ^ {4{lfsr}}

Trojan5 - Program Counter Redirection
- Description: Alters control flow by changing program address output when instruction type matches specific opcodes.
- Behavior: Control flow manipulation
- Trigger: Specific prog_dat_i[13:10] patterns
- Output: prog_adr_o = pc_reg + 2 or pc_reg

Trojan6 - Triggered Payload Insertion Based on Magic Constants
- Description: Detects exact 32-bit values and sets payload bits if matched.
- Behavior: Control logic manipulation
- Trigger: m0_data_o == specific constants
- Output: Modified i_s15_data_o_TrojanPayload

Trojan7 - Selective Slave Output Modification
- Description: If both wb_data_i and s0_data_i match hardcoded values, modifies address select signal.
- Behavior: Selective logic modification
- Trigger: Exact value match
- Output: slv_sel = wb_addr_i[31:28] ^ 0xF or direct

Trojan8 - Complex Combinational Data Obfuscation
- Description: Performs multiple multiply-add/XOR/bitmask operations, heavily obscuring output.
- Behavior: Logic obfuscation / functional pollution
- Trigger: Input-driven (sel)
- Output: y ← t1–t7, based on selection

Trojan9 - Multi-mode Arithmetic Logic Unit Manipulation
- Description: Performs different arithmetic logic per mode with final output modified through XOR and shifts.
- Behavior: Logic alteration
- Trigger: mode selector
- Output: y ← m1–m4 depending on mode

Note:
- All Trojans fall within the 5 behavior classes: Information Leakage, Trigger Events, Control Flow, Logic Modification, Obfuscation.
- Some contain hardcoded patterns, making generalization crucial during training.
