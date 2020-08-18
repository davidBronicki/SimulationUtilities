namespace SimulationUtilities{

	namespace
	{
		//raise base to exponent power at compile time

		template<size_t base, size_t exponent>
		struct Template_Power : public std::integral_constant<size_t, base * Template_Power<base, exponent - 1>::value>{};

		template<size_t base>
		struct Template_Power<base, 0> : public std::integral_constant<size_t, 1>{};

		//Template_Locate_Nth_Key_Type value is the location of the positionth instance
		//of Key in Pack... (position and value are zero indexed)

		template<size_t position, typename Key, typename... Pack>
		struct Template_Locate_Nth_Key_Type : public std::integral_constant<size_t, sizeof...(Pack)>{};

		template<size_t position, typename Key, typename Next, typename... Pack>
		struct Template_Locate_Nth_Key_Type<position, Key, Next, Pack...> :
			public std::integral_constant<size_t, 1 + Template_Locate_Nth_Key_Type<position, Key, Pack...>::value>{};

		template<size_t position, typename Key, typename... Pack>
		struct Template_Locate_Nth_Key_Type<position, Key, Key, Pack...> :
			public std::integral_constant<size_t, 1 + Template_Locate_Nth_Key_Type<position - 1, Key, Pack...>::value>{};

		template<typename Key, typename... Pack>
		struct Template_Locate_Nth_Key_Type<0, Key, Key, Pack...> :
			public std::integral_constant<size_t, 0>{};

		//Template_Locate_Key_Type value is the location of the first instance of
		//Key in Pack... (directly based on nth version)

		template<typename Key, typename... Pack>
		struct Template_Locate_Key_Type : public Template_Locate_Nth_Key_Type<0, Key, Pack...>{};

		//Template_Key_In_Pack value is true if Key is found in Pack..., false otherwise.

		template<typename Key, typename... Pack>
		struct Template_Key_In_Pack;

		template<typename Key, typename Next, typename... Pack>
		struct Template_Key_In_Pack<Key, Next, Pack...> : public Template_Key_In_Pack<Key, Pack...>{};

		template<typename Key, typename... Pack>
		struct Template_Key_In_Pack<Key, Key, Pack...>
		{
			static constexpr bool value = true;
		};

		template<typename Key>
		struct Template_Key_In_Pack<Key>
		{
			static constexpr bool value = false;
		};

		//generic indexed tensor type

		template<size_t rank, size_t dimensions, typename T, typename... indexIdentifiers>
		struct IndexedTensor
		{
			static_assert(sizeof...(indexIdentifiers)==rank, "Invalid number of indices on tensor.");
		};

		//type to store a group of index type differentiators

		template<typename... Is>
		struct IndexPackType{};

		//Template_Pack_If T is empty IndexPackType if packCondition is false
		//and is IndexPackType<Key> if packConidition is true

		template<typename Key, bool packCondition>
		struct Template_Pack_If
		{
			typedef IndexPackType<> T;
		};

		template<typename Key>
		struct Template_Pack_If<Key, true>
		{
			typedef IndexPackType<Key> T;
		};

		//Template_Condense smashes two IndexPackTypes into one

		template<typename First, typename Last>
		struct Template_Condense
		{
			static_assert(std::is_same<First,Last>::value, "Template_Condense accepts only IndexPackType template arguments.");
			static_assert(!std::is_same<First,Last>::value, "Template_Condense accepts only IndexPackType template arguments.");
		};

		template<typename... Pack1, typename... Pack2>
		struct Template_Condense<IndexPackType<Pack1...>, IndexPackType<Pack2...>>{typedef IndexPackType<Pack1..., Pack2...> T;};

		//Template_Get_Repeats creates an IndexPackType holding repeat types in Is...

		template<typename... Is>
		struct Template_Get_Repeats
		{
			typedef IndexPackType<> T;
		};

		template<typename Next, typename... Others>
		struct Template_Get_Repeats<Next, Others...>
		{
			typedef typename Template_Condense<
				typename Template_Pack_If<Next, Template_Key_In_Pack<Next, Others...>::value>::T,
					typename Template_Get_Repeats<Others...>::T>::T T;
		};

		//Template_Remove_Type removes all instances of Key from Pack and gives an IndexPackType with the new pack
		//or removes all in first IndexPackType from second IndexPackType

		template<typename Key, typename... Pack>
		struct Template_Remove_Type
		{
			typedef IndexPackType<> T;
		};

		template<typename Key, typename Next, typename... Pack>
		struct Template_Remove_Type<Key, Next, Pack...>
		{
			typedef	typename Template_Condense<
				typename Template_Pack_If<Next, !std::is_same<Key, Next>::value>::T,
				typename Template_Remove_Type<Key, Pack...>::T>::T T;
		};

		template<typename... Keys, typename... Pack>
		struct Template_Remove_Type<IndexPackType<Keys...>, IndexPackType<Pack...>>
		{
			typedef IndexPackType<Pack...> T;
		};

		template<typename NextKey, typename... OtherKeys, typename... Pack>
		struct Template_Remove_Type<IndexPackType<NextKey, OtherKeys...>, IndexPackType<Pack...>>
		{
			typedef typename Template_Remove_Type<IndexPackType<OtherKeys...>,
				typename Template_Remove_Type<NextKey, Pack...>::T>::T T;
		};

		//Template_Remove_Repeats gives an IndexPackType with all repeats removed.

		template<typename... Is>
		struct Template_Remove_Repeats
		{
			typedef typename Template_Remove_Type<typename Template_Get_Repeats<Is...>::T, IndexPackType<Is...>>::T T;
		};

		template<typename a, typename b>
		struct Template_Equal_Packs;

		template<typename... Pack1, typename... Pack2>
		struct Template_Equal_Packs<IndexPackType<Pack1...>, IndexPackType<Pack2...>>
		{
			static constexpr bool value = false;
		};

		template<typename Next, typename... Pack1, typename... Pack2>
		struct Template_Equal_Packs<IndexPackType<Next, Pack1...>, IndexPackType<Pack2...>>
		{
			static constexpr bool value = Template_Key_In_Pack<Next, Pack2...>::value &&
				Template_Equal_Packs<IndexPackType<Pack1...>, typename Template_Remove_Type<Next, Pack2...>::T>::value;
		};

		template<>
		struct Template_Equal_Packs<IndexPackType<>, IndexPackType<>>
		{
			static constexpr bool value = true;
		};
	}

}