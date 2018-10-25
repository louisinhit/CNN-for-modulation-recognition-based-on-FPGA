library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use std.textio.all;


entity mynet_layer is
  generic (
    input_feature_width         : natural;
    input_no_feature_planes_par : natural;
    input_no_feature_planes_ser : natural;
    input_feature_plane_width   : natural;
    input_feature_plane_height  : natural;
    zero_pad_top                : natural;
    zero_pad_bottom             : natural;
    zero_pad_left               : natural;
    zero_pad_right              : natural;
    input_mask_width            : natural;
    input_mask_height           : natural;
    input_stride                : natural;
    narrow_buffer_depth         : natural;
    narrow_accept_space         : natural;
    no_par_layers               : natural;
    layer_size                  : natural;
    layer_size_order            : natural;
    weight_width                : natural;
    weight_mem_order            : natural;
    output_width                : natural;
    output_shift                : natural;
    ReLU                        : boolean;
    output_par_widen_factor     : natural;
    use_maxpool                 : boolean;
    maxpool_mask_width          : natural;
    maxpool_mask_height         : natural;
    maxpool_stride              : natural);   
  port (
    clk                  : in  std_logic;
    rst                  : in  std_logic;
    input_feature_stream : in  std_logic_vector(input_feature_width*input_no_feature_planes_par-1 downto 0);
    input_feature_valid  : in  std_logic;
    input_feature_ready  : out std_logic;
    weight_stream        : in  std_logic_vector(weight_width-1 downto 0);
    weight_id            : in  std_logic_vector(layer_size_order-1 downto 0);
    weight_first         : in  std_logic_vector(no_par_layers-1 downto 0);
    weight_last          : in  std_logic_vector(no_par_layers-1 downto 0);

    stream_out       : out std_logic_vector(output_width*output_par_widen_factor*no_par_layers-1 downto 0);
    stream_out_valid : out std_logic;
    stream_out_ready : in  std_logic
    );

end entity;


--------------------------------------------
architecture rtl of mynet_layer is

   constant zp_feature_plane_width   : natural := input_feature_plane_width+zero_pad_left+zero_pad_right;
   constant zp_feature_plane_height  : natural := input_feature_plane_height+zero_pad_top+zero_pad_bottom;


   constant maxpool_feature_plane_width : natural := (zp_feature_plane_width-input_mask_width)/input_stride;
   constant maxpool_feature_plane_height : natural := (zp_feature_plane_height-input_mask_height)/input_stride;

   constant zp_maxpool_pad : natural := maxpool_mask_width-maxpool_stride;
  
  component zero_pad_stream is
    generic (
      stream_width : natural;
      zero_pad_top : natural;
      zero_pad_bottom : natural;
      zero_pad_left : natural;
      zero_pad_right : natural;
      input_height : natural;
      input_width  : natural);
    port (
      clk              : in  std_logic;
      rst              : in  std_logic;
      stream_in        : in  std_logic_vector(stream_width-1 downto 0);
      stream_in_valid  : in  std_logic;
      stream_in_ready  : out std_logic;
      stream_out       : out std_logic_vector(stream_width-1 downto 0);
      stream_out_valid : out std_logic;
      stream_out_ready : in  std_logic);
  end component zero_pad_stream;
  
  component feature_buffer is
    generic (
      feature_width         : natural;
      no_feature_planes_par : natural;
      no_feature_planes_ser : natural;
      mask_width            : natural;
      mask_height           : natural;
      feature_plane_width   : natural;
      feature_plane_height  : natural;
      stride                : natural);
    port (
      clk                 : in  std_logic;
      rst                 : in  std_logic;
      feature_stream      : in  std_logic_vector(feature_width*no_feature_planes_par-1 downto 0);
      feature_valid       : in  std_logic;
      feature_ready       : out std_logic;
      mask_feature_stream : out std_logic_vector(feature_width*no_feature_planes_par-1 downto 0);
      mask_feature_valid  : out std_logic;
      mask_feature_ready  : in  std_logic;
      mask_feature_first  : out std_logic;
      mask_feature_last   : out std_logic);
  end component feature_buffer;

  component stream_narrow is
    generic (
      stream_width         : natural;
      stream_in_multiplier : natural;
      buffer_depth         : natural;
      buffer_accept_space  : natural;
      burst_level : natural);
    port (
      clk              : in  std_logic;
      rst              : in  std_logic;
      stream_in        : in  std_logic_vector(stream_width*stream_in_multiplier-1 downto 0);
      stream_in_valid  : in  std_logic;
      stream_in_first  : in  std_logic;
      stream_in_last   : in  std_logic;
      stream_in_ready  : out std_logic;
      stream_out       : out std_logic_vector(stream_width-1 downto 0);
      stream_out_valid : out std_logic;
      stream_out_first : out std_logic;
      stream_out_last  : out std_logic);
  end component stream_narrow;

  component conv_neuron_layer_ffanout is
    generic (
      layer_size       : natural;
      layer_size_order : natural;
      feature_width    : natural;
      weight_width     : natural;
      weight_mem_order : natural;
      output_width     : natural;
      output_shift     : natural;
      bias_shift       : natural;
      ReLU             : boolean);
    port (
      clk            : in  std_logic;
      feature_stream : in  std_logic_vector(feature_width-1 downto 0);
      feature_first  : in  std_logic;
      feature_last   : in  std_logic;
      weight_stream  : in  std_logic_vector(weight_width-1 downto 0);
      weight_id      : in  std_logic_vector(layer_size_order-1 downto 0);
      weight_first   : in  std_logic;
      weight_last    : in  std_logic;
      output_stream  : out std_logic_vector(output_width-1 downto 0);
      output_id      : out std_logic_vector(layer_size_order-1 downto 0);
      output_valid   : out std_logic);
  end component conv_neuron_layer_ffanout;

  component stream_widen is
    generic (
      stream_width          : natural;
      stream_out_multiplier : natural);
    port (
      clk              : in  std_logic;
      rst              : in  std_logic;
      stream_in        : in  std_logic_vector(stream_width-1 downto 0);
      stream_in_valid  : in  std_logic;
      stream_in_first  : in  std_logic;
      stream_in_last   : in  std_logic;
      stream_out       : out std_logic_vector(stream_width*stream_out_multiplier-1 downto 0);
      stream_out_valid : out std_logic;
      stream_out_first : out std_logic;
      stream_out_last  : out std_logic);
  end component stream_widen;


  component maxpool is
    generic (
      feature_width         : natural;
      no_feature_planes_par : natural;
      no_feature_planes_ser : natural;
      pool_size             : natural);
    port (
      clk                : in  std_logic;
      rst                : in  std_logic;
      feature_stream     : in  std_logic_vector(feature_width*no_feature_planes_par-1 downto 0);
      feature_valid      : in  std_logic;
      feature_first      : in  std_logic;
      feature_last       : in  std_logic;
      max_feature_stream : out std_logic_vector(feature_width*no_feature_planes_par-1 downto 0);
      max_feature_valid  : out std_logic);
  end component maxpool;
   
   component zero_pad_stream_mp is
     generic (
       stream_width    : natural;
       zero_pad_bottom : natural;
       zero_pad_right  : natural;
       input_height    : natural;
       input_width     : natural);
     port (
       clk              : in  std_logic;
       rst              : in  std_logic;
       stream_in        : in  std_logic_vector(stream_width-1 downto 0);
       stream_in_valid  : in  std_logic;
       stream_in_ready  : out std_logic;
       stream_out       : out std_logic_vector(stream_width-1 downto 0);
       stream_out_valid : out std_logic;
       stream_out_ready : in  std_logic);
   end component zero_pad_stream_mp;
   
  signal mask_feature_stream : std_logic_vector(input_feature_width*input_no_feature_planes_par-1 downto 0);
  signal mask_feature_valid  : std_logic;
  signal mask_feature_ready  : std_logic;
  signal mask_feature_first  : std_logic;
  signal mask_feature_last   : std_logic;

  signal narrow_feature_stream : std_logic_vector(input_feature_width-1 downto 0);
  signal narrow_feature_first : std_logic;
  signal narrow_feature_last  : std_logic;
   signal narrow_feature_valid  : std_logic;


  signal widen_stream_out       : std_logic_vector(output_width*output_par_widen_factor*no_par_layers-1 downto 0);
  signal widen_stream_out_valid : std_logic;
  signal widen_stream_out_first : std_logic;
  signal widen_stream_out_last  : std_logic;

  signal widen_stream_out1       : std_logic_vector(output_width*output_par_widen_factor*no_par_layers-1 downto 0);
  signal widen_stream_out_valid1 : std_logic;
   
  
  signal maxpool_ready : std_logic;

  signal maxpool_feature_stream : std_logic_vector(output_width*output_par_widen_factor*no_par_layers-1 downto 0);
  signal maxpool_feature_valid  : std_logic;
  signal maxpool_feature_first  : std_logic;
  signal maxpool_feature_ready  : std_logic;
  signal maxpool_feature_last   : std_logic;
  signal max_feature_stream     : std_logic_vector(output_width*output_par_widen_factor*no_par_layers-1 downto 0);
  signal max_feature_valid      : std_logic;

  signal zp_feature_stream : std_logic_vector(input_feature_width*input_no_feature_planes_par-1 downto 0);
  signal zp_feature_valid  : std_logic;
  signal zp_feature_ready  : std_logic;

   signal layer_output_first, layer_output_last : std_logic;
   signal layer_output_id       : std_logic_vector(layer_size_order*no_par_layers-1 downto 0);
   signal layer_output_stream       : std_logic_vector(output_width*no_par_layers-1 downto 0);
   signal layer_output_valid       : std_logic_vector(no_par_layers-1 downto 0);

   signal input_feature_ready_i : std_logic := '0';
   
   constant debug : boolean := true;
   
begin

  gen_zero_pad: if zero_pad_left >0 or zero_pad_right >0 or zero_pad_top >0 or zero_pad_bottom >0 generate
    zero_pad_stream_1: zero_pad_stream
      generic map (
        stream_width => input_feature_width*input_no_feature_planes_par,
        zero_pad_top => zero_pad_top,
        zero_pad_bottom => zero_pad_bottom,
        zero_pad_left => zero_pad_left,
        zero_pad_right => zero_pad_right,
        input_height => input_feature_plane_height,
        input_width  => input_feature_plane_width)
      port map (
        clk              => clk,
        rst              => rst,
        stream_in        => input_feature_stream,
        stream_in_valid  => input_feature_valid,
        stream_in_ready  => input_feature_ready_i,
        stream_out       => zp_feature_stream,
        stream_out_valid => zp_feature_valid,
        stream_out_ready => zp_feature_ready);

    input_feature_ready <= input_feature_ready_i;
  end generate;

  gen_no_zero_pad: if zero_pad_left=0 and zero_pad_right =0 and zero_pad_top =0 and zero_pad_bottom =0 generate
    zp_feature_stream <= input_feature_stream;
    zp_feature_valid <= input_feature_valid;
    input_feature_ready <= zp_feature_ready;
  end generate;
  

  
  feature_buffer_1 : feature_buffer
    generic map (
      feature_width         => input_feature_width,
      no_feature_planes_par => input_no_feature_planes_par,
      no_feature_planes_ser => input_no_feature_planes_ser,
      mask_width            => input_mask_width,
      mask_height           => input_mask_height,
      feature_plane_width   => zp_feature_plane_width,
      feature_plane_height  => zp_feature_plane_height,
      stride                => input_stride)
    port map (
      clk                 => clk,
      rst                 => rst,
      feature_stream      => zp_feature_stream,
      feature_valid       => zp_feature_valid,
      feature_ready       => zp_feature_ready,
      mask_feature_stream => mask_feature_stream,
      mask_feature_valid  => mask_feature_valid,
      mask_feature_ready  => mask_feature_ready,
      mask_feature_first  => mask_feature_first,
      mask_feature_last   => mask_feature_last);


  -- Narrow stream to generate a continuous stream
  -- at neuron layer input
  
  stream_narrow_1 : stream_narrow
    generic map (
      stream_width         => input_feature_width,
      stream_in_multiplier => input_no_feature_planes_par/no_par_layers,
      buffer_depth         => narrow_buffer_depth,
      buffer_accept_space  => narrow_accept_space,
      burst_level => layer_size/(input_no_feature_planes_par/no_par_layers))
    port map (
      clk              => clk,
      rst              => rst,
      stream_in        => mask_feature_stream,
      stream_in_valid  => mask_feature_valid,
      stream_in_first  => mask_feature_first,
      stream_in_last   => mask_feature_last,
      stream_in_ready  => mask_feature_ready,
      stream_out       => narrow_feature_stream,
      stream_out_valid => narrow_feature_valid,
      stream_out_first => narrow_feature_first,
      stream_out_last  => narrow_feature_last);



  gen_layers : for i in 0 to no_par_layers-1 generate
    conv_neuron_1 : conv_neuron_layer_ffanout
      generic map (
        layer_size       => layer_size,
        layer_size_order => layer_size_order,
        feature_width    => input_feature_width,
        weight_width     => weight_width,
        weight_mem_order => weight_mem_order,
        output_width     => output_width,
        output_shift     => output_shift,
        bias_shift       => 0,
        ReLU             => ReLU)
      port map (
        clk            => clk,
        feature_stream => narrow_feature_stream,
        feature_first  => narrow_feature_first,
        feature_last   => narrow_feature_last,
        weight_stream  => weight_stream,
        weight_id      => weight_id,
        weight_first   => weight_first(i),
        weight_last    => weight_last(i),
        output_stream  => layer_output_stream(output_width*(i+1)-1 downto output_width*i),
        output_id      => layer_output_id(layer_size_order*(i+1)-1 downto layer_size_order*i),
        output_valid   => layer_output_valid(i));
  end generate;

  layer_output_first <= '1' when to_integer(unsigned(layer_output_id(layer_size_order-1 downto 0))) = 1          else '0';
  layer_output_last  <= '1' when to_integer(unsigned(layer_output_id(layer_size_order-1 downto 0))) = layer_size else '0';



  stream_widen_1 : stream_widen
    generic map (
      stream_width          => output_width*no_par_layers,
      stream_out_multiplier => output_par_widen_factor)
    port map (
      clk              => clk,
      rst              => rst,
      stream_in        => layer_output_stream,
      stream_in_valid  => layer_output_valid(0),
      stream_in_first  => layer_output_first,
      stream_in_last   => layer_output_last,
      stream_out       => widen_stream_out,
      stream_out_valid => widen_stream_out_valid,
      stream_out_first => widen_stream_out_first,
      stream_out_last  => widen_stream_out_last);


  gen_no_maxpool : if not use_maxpool generate
    stream_out       <= widen_stream_out;
    stream_out_valid <= widen_stream_out_valid;

    assert (widen_stream_out_valid /= '1' or stream_out_ready /= '0') report "Tried to output data when data sink not ready" severity failure;
    
  end generate;

  gen_maxpool : if use_maxpool generate
    

    assert (widen_stream_out_valid1 /= '1' or maxpool_ready /= '0') report "Tried to push data into full maxpool buffer, please check layer structure" severity failure;

    assert (max_feature_valid /= '1' or stream_out_ready/= '0') report "Tried to output data when data sink not ready" severity failure;

    gen_mp_pad: if zp_maxpool_pad >0 generate
    
    zero_pad_stream_mp_1: zero_pad_stream_mp
      generic map (
        stream_width    => output_width*no_par_layers*output_par_widen_factor,
        zero_pad_bottom => zp_maxpool_pad,
        zero_pad_right  => zp_maxpool_pad*(layer_size/output_par_widen_factor),
        input_height    => maxpool_feature_plane_height,
        input_width     => maxpool_feature_plane_width*(layer_size/output_par_widen_factor) )
      port map (
        clk              => clk,
        rst              => rst,
        stream_in        => widen_stream_out,
        stream_in_valid  => widen_stream_out_valid,
        stream_in_ready  => open,
        stream_out       => widen_stream_out1,
        stream_out_valid => widen_stream_out_valid1,
        stream_out_ready => maxpool_ready);
    end generate;

    gen_no_mp_pad: if zp_maxpool_pad =0 generate  
      widen_stream_out1 <= widen_stream_out;
      widen_stream_out_valid1 <= widen_stream_out_valid;   
    end generate;
   
    maxpool_buffer_1 : feature_buffer
      generic map (
        feature_width         => output_width,
        no_feature_planes_par => no_par_layers*output_par_widen_factor,
        no_feature_planes_ser => layer_size/output_par_widen_factor,
        mask_width            => maxpool_mask_width,
        mask_height           => maxpool_mask_height,
        feature_plane_width   => maxpool_feature_plane_width+zp_maxpool_pad,
        feature_plane_height  => maxpool_feature_plane_height+zp_maxpool_pad,
        stride                => maxpool_stride)
      port map (
        clk                 => clk,
        rst                 => rst,
        feature_stream      => widen_stream_out1,
        feature_valid       => widen_stream_out_valid1,
        feature_ready       => maxpool_ready,
        mask_feature_stream => maxpool_feature_stream,
        mask_feature_valid  => maxpool_feature_valid,
        mask_feature_ready  => maxpool_feature_ready,
        mask_feature_first  => maxpool_feature_first,
        mask_feature_last   => maxpool_feature_last);

    maxpool_feature_ready <= '1';

    maxpool_1 : maxpool
      generic map (
        feature_width         => output_width,
        no_feature_planes_par => no_par_layers*output_par_widen_factor,
        no_feature_planes_ser => layer_size/output_par_widen_factor,
        pool_size             => maxpool_mask_width*maxpool_mask_height)
      port map (
        clk                => clk,
        rst                => rst,
        feature_stream     => maxpool_feature_stream,
        feature_valid      => maxpool_feature_valid,
        feature_first      => maxpool_feature_first,
        feature_last       => maxpool_feature_last,
        max_feature_stream => max_feature_stream,
        max_feature_valid  => max_feature_valid);

    stream_out       <= max_feature_stream;
    stream_out_valid <= max_feature_valid;
  end generate;

-- synthesis translate off

 -- gen_dbg: if debug generate
  
  debug_log0: process(clk) is
     variable l : line;
     variable count : integer := 0;
     variable licount : integer := 0;
     variable mpcount : integer := 0;
     variable zpcount : integer := 0;
     variable ipcount : integer := 0;
     variable zplcount : integer := 0;
  begin
    if rising_edge(clk) then
      count := count+1;
      if false then
      write (l,count);
      count := count+1;
      write (l, string'(": "));
      write (l, string'("IF"));
      if input_feature_valid = '1' then
        write (l, string'("V"));
      else
        write (l, string'("."));
      end if;
       write (l, string'(" ZF"));
      if zp_feature_valid = '1' then
        write (l, string'("V"));
      else
        write (l, string'("."));
      end if;
      if zp_feature_ready = '1' then
        write (l, string'("R"));
      else
        write (l, string'("."));
      end if;
      write (l, string'(" MF"));
      if mask_feature_valid = '1' then
        write (l, string'("V"));
      else
        write (l, string'("."));
      end if;
      if mask_feature_ready = '1' then
        write (l, string'("R"));
      else
        write (l, string'("."));
      end if;
      if mask_feature_first = '1' then
        write (l, string'("F"));
      else
        write (l, string'("."));
      end if;
      if mask_feature_last = '1' then
        write (l, string'("L"));
      else
        write (l, string'("."));
      end if;
      write (l, string'(" NF"));
      if narrow_feature_first = '1' then
        write (l, string'("F"));
      else
        write (l, string'("."));
      end if;
      if narrow_feature_last = '1' then
        write (l, string'("L"));
      else
        write (l, string'("."));
      end if;
      write (l, string'(" LO#"));
      write (l, to_integer(unsigned(layer_output_id(layer_size_order-1 downto 0))));
      if layer_output_first = '1' then
        write (l, string'("F"));
      else
        write (l, string'("."));
      end if;
      if layer_output_last = '1' then
        write (l, string'("L"));
      else
        write (l, string'("."));
      end if;

      write (l, string'(" WO"));
       if widen_stream_out_valid = '1' then
        write (l, string'("V"));
      else
        write (l, string'("."));
      end if;
      if widen_stream_out_first = '1' then
        write (l, string'("F"));
      else
        write (l, string'("."));
      end if;
      if widen_stream_out_last = '1' then
        write (l, string'("L"));
      else
        write (l, string'("."));
      end if;

      write (l, string'(" MP"));
      if maxpool_feature_valid = '1' then
        write (l, string'("V"));
      else
        write (l, string'("."));
      end if;
      if maxpool_feature_first = '1' then
        write (l, string'("F"));
      else
        write (l, string'("."));
      end if;
      if maxpool_feature_last = '1' then
        write (l, string'("L"));
      else
        write (l, string'("."));
      end if;
      if maxpool_feature_ready = '1' then
        write (l, string'("R"));
      else
        write (l, string'("."));
      end if;
      if maxpool_ready = '1' then
        write (l, string'("M"));
      else
        write (l, string'("."));
      end if;
      end if;

      --if narrow_feature_first = '1' then
      --  if (licount mod ((zp_feature_plane_width-(input_mask_width-1)/2)/input_stride)) = 0 then
      --    write (l, string'("Start of Layer Input: Cycle: "));
      --    write (l,count);
      --    write (l,string'(" #"));
      --    write (l,licount);
          
      --    writeline(output,l);
      --  end if;
      --  licount := licount+1; 
      --end if;

      --if maxpool_feature_first = '1' and maxpool_feature_valid = '1' then
      --  if (mpcount mod ((maxpool_feature_plane_width-(maxpool_mask_width-1)/2)/maxpool_stride)) = 0 then
      --    write (l, string'("Start of Maxpool Input: Cycle: "));
      --    write (l,count);
      --    write (l,string'(" #"));
      --    write (l,mpcount);

      --    write (l,string'(" Pixel #"));
      --    write (l,zpcount);
      --    write (l,string'(" IPixel #"));
      --    write (l,ipcount);
      --    writeline(output,l);
      --  end if;
      --  mpcount := mpcount+1; 
      --end if;
      --if input_feature_valid ='1' and input_feature_ready_i = '1' then
      --  ipcount := ipcount+1;
      --end if;
      --if zp_feature_valid = '1' and zp_feature_ready = '1' then
      --  if (zpcount mod zp_feature_plane_width) = zp_feature_plane_width-1 then
      --    write (l, string'("End of ZP Line: Cycle: "));
      --    write (l,count);
      --    write (l,string'(" Pixel #"));
      --    write (l,zpcount);
      --    write (l,string'(" IPixel #"));
      --    write (l,ipcount);
      --    write (l,string'(" Line #"));
      --    write (l,zplcount);
      --    write (l,string'(" Pos #"));
      --    write (l,zpcount mod zp_feature_plane_width);
      --    writeline(output,l);
      --    zplcount := zplcount+1;
      --  end if;
      --  zpcount := zpcount+1;
      --end if;

      --if narrow_feature_valid = '1' then
      --   write (l,to_integer(unsigned(narrow_feature_stream)));
      --    write (l,string'(" "));
      --   if narrow_feature_last = '1' then
      --     writeline(output,l);
      --   end if;
      --end if;  
    end if;
  end process;
  
 -- end generate;
-- synthesis translate_on  
  
end architecture;
